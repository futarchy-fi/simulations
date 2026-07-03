"""LLM-backed decision-market agent.

Calls the local `claude` CLI once per acting opportunity. The agent is told
the full game honestly (signal, precision, importance, wealth, costs, payout
rules) and asked for strict JSON:

    {"belief_x_positive": float 0-1, "action": "stake_yes"|"stake_no"|"pass",
     "amount": float}

Round-0 prompts deliberately exclude all market state so the logged
first-round belief is a clean private-information "poll" answer (Arm C).
Later rounds include the published market state.

Every call (prompt, raw response, parsed output, latency) is appended to a
JSONL log file for auditability.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

from proposal_poker.interfaces import AgentBase
from proposal_poker.types import Contribution

_SYSTEM_PROMPT = (
    "You are an expert quantitative trading agent playing a decision market. "
    "You always reply with exactly one raw JSON object and nothing else: "
    "no code fences, no prose, no explanation."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMMarketAgent(AgentBase):
    agent_id = "llm_market"

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        log_dir: str | None = None,
        shard_id: str = "s0",
        phi: float = 0.01,
        fee_rate: float = 0.01,
        precision_ratio: float = 2.0,
        max_rounds: int = 3,
        call_timeout_s: float = 180.0,
        connection_retries: int = 3,
        connection_retry_wait_s: float = 60.0,
        **params: object,
    ) -> None:
        super().__init__(**params)
        self.model = model
        self.shard_id = shard_id
        self.phi = float(phi)
        self.fee_rate = float(fee_rate)
        self.precision_ratio = float(precision_ratio)
        self.max_rounds = int(max_rounds)
        self.call_timeout_s = float(call_timeout_s)
        self.connection_retries = int(connection_retries)
        self.connection_retry_wait_s = float(connection_retry_wait_s)
        self.instance_uid = uuid.uuid4().hex[:8]
        self._proposal_counter = -1

        log_root = log_dir or os.environ.get("LLM_DM_LOG_DIR") or "."
        Path(log_root).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_root) / f"calls_{shard_id}_{self.instance_uid}.jsonl"

    # ------------------------------------------------------------------ act

    def act(
        self,
        wealth: float,
        signal: float,
        y: float,
        public_history: list[object],
        my_past: list[Contribution],
    ) -> Contribution | None:
        round_index = self._round_index(public_history)
        if round_index == 0:
            self._proposal_counter += 1

        prompt = self._build_prompt(
            wealth=wealth,
            signal=signal,
            y=y,
            round_index=round_index,
            public_history=public_history,
            my_past=my_past,
        )

        parsed, raw, latency, error = self._call_llm(prompt)
        record = {
            "ts": time.time(),
            **self._log_extra(),
            "shard_id": self.shard_id,
            "instance_uid": self.instance_uid,
            "model": self.model,
            "proposal_local_index": self._proposal_counter,
            "round_index": round_index,
            "wealth": wealth,
            "signal": signal,
            "y": y,
            "prompt": prompt,
            "raw_response": raw,
            "parsed": parsed,
            "latency_s": latency,
            "error": error,
        }

        contribution = None
        if parsed is not None:
            contribution = self._to_contribution(parsed, wealth, y, my_past)
            record["contribution"] = (
                None
                if contribution is None
                else {"amount": contribution.amount, "data": contribution.data}
            )
        self._log(record)
        return contribution

    # -------------------------------------------------------------- helpers

    def _log_extra(self) -> dict:
        return {"agent_type": self.agent_id}

    def _round_index(self, public_history: list[object]) -> int:
        if not public_history:
            return 0
        last = public_history[-1]
        if isinstance(last, dict) and "round" in last:
            try:
                return int(last["round"])
            except (TypeError, ValueError):
                return 0
        return 0

    def _build_prompt(
        self,
        wealth: float,
        signal: float,
        y: float,
        round_index: int,
        public_history: list[object],
        my_past: list[Contribution],
    ) -> str:
        tau = self.precision_ratio * wealth
        noise_std = 1.0 / math.sqrt(tau)
        entry_cost = self.phi * wealth * math.sqrt(y)
        subsidy = self._winner_subsidy(public_history)

        lines = [
            "DECISION MARKET GAME. A proposal has hidden quality x drawn from a standard normal distribution N(0,1).",
            "The proposal should be approved if and only if x > 0. You profit by staking on the side that ends up being the final decision.",
            "",
            "MARKET RULES (pari-mutuel binary staking):",
            "- You may stake money on YES (approve) or NO (reject). Other agents do too.",
            "- After the final round, the side with more total stake wins the decision (NO wins exact ties).",
            "- EXCEPTION: if the losing margin is small, |YES-NO|/total < 10%, a noisy verification oracle is invoked: oracle reading z = x + noise (noise std 0.316), decision becomes approve iff z > 0, and z determines the winning side.",
            f"- Settlement: the pot = all YES + NO stakes PLUS a public sponsor subsidy of {subsidy:.2f}, split pro-rata among stakes on the winning side. If you staked amount a on the winning side, you receive a * (total_stakes + {subsidy:.2f}) / winning_side_total. Losing stakes get nothing. The subsidy is paid only if someone staked.",
            "",
            "YOUR PRIVATE INFORMATION AND COSTS:",
            f"- Your private signal: s = {signal:.4f}. It equals x plus Gaussian noise with std {noise_std:.4f} (precision {tau:.3f}). Signals of other agents are independent given x; wealthier agents have more precise signals.",
            f"- Proposal importance y = {y:.4f} (public; scales your participation cost, not your payout).",
            f"- Your wealth W = {wealth:.4f}.",
            f"- One-time participation cost if you stake anything on this proposal: {entry_cost:.4f} (deadweight, charged once on your first accepted stake).",
            f"- Stake fee: {self.fee_rate:.0%} of every unit you stake (deadweight).",
            "- Your utility is log(terminal wealth) - log(W). Not participating gives exactly 0. Only stake when your edge covers the costs; size stakes like a log-utility (Kelly-style) bettor.",
        ]

        if round_index == 0:
            lines += [
                "",
                f"This is acting round 1 of {self.max_rounds}. No market information is available to you yet. You will see the market state and may add stakes in later rounds.",
            ]
        else:
            approve, reject = self._market_state(public_history)
            past = [
                {"side": c.data.get("side"), "amount": round(c.amount, 4)}
                for c in my_past
            ]
            paid = "already paid" if my_past else "not yet paid (charged if you stake now)"
            lines += [
                "",
                f"This is acting round {round_index + 1} of {self.max_rounds}.",
                f"CURRENT MARKET STATE: YES pool = {approve:.4f}, NO pool = {reject:.4f} (includes any of your own past stakes).",
                f"Your past stakes on this proposal: {json.dumps(past)}. Participation cost: {paid}.",
                "You may add stake to either side (stakes are additive and irreversible) or pass.",
            ]

        lines += [
            "",
            'Respond with ONLY one raw JSON object, no code fences, no other text:',
            '{"belief_x_positive": <your probability 0-1 that x > 0>, "action": "stake_yes"|"stake_no"|"pass", "amount": <money to stake now, 0 if pass>}',
        ]
        return "\n".join(lines)

    def _winner_subsidy(self, public_history: list[object]) -> float:
        for message in reversed(public_history):
            if isinstance(message, dict) and "winner_subsidy" in message:
                return float(message["winner_subsidy"])
        return 0.0

    def _market_state(self, public_history: list[object]) -> tuple[float, float]:
        for message in reversed(public_history):
            if isinstance(message, dict) and "approve_stake" in message:
                return (
                    float(message.get("approve_stake", 0.0)),
                    float(message.get("reject_stake", 0.0)),
                )
        return 0.0, 0.0

    # ------------------------------------------------------------- LLM call

    def _call_llm(self, prompt: str):
        raw_all: list[str] = []
        started = time.perf_counter()
        error = None
        parsed = None

        for attempt in range(2):  # parse retry once
            raw, call_error = self._invoke_cli(prompt if attempt == 0 else prompt + "\n\nREMINDER: output ONLY the raw JSON object.")
            raw_all.append(raw if call_error is None else f"<{call_error}>")
            if call_error is not None:
                error = call_error
                break
            parsed = self._parse(raw)
            if parsed is not None:
                error = None
                break
            error = "parse_failure"

        latency = time.perf_counter() - started
        return parsed, "\n---RETRY---\n".join(raw_all), latency, error

    def _invoke_cli(self, prompt: str) -> tuple[str, str | None]:
        cmd = [
            "claude",
            "-p",
            prompt,
            "--model",
            self.model,
            "--strict-mcp-config",
            "--disallowedTools",
            "*",
            "--system-prompt",
            _SYSTEM_PROMPT,
        ]
        env = dict(os.environ)
        # Bound extended thinking; unbounded thinking made haiku calls take
        # 80-140s each vs ~17s with a 1024-token budget.
        env["MAX_THINKING_TOKENS"] = os.environ.get("LLM_DM_THINKING_TOKENS", "1024")

        output = ""
        error = "no_attempts"
        for attempt in range(self.connection_retries + 1):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.call_timeout_s,
                    stdin=subprocess.DEVNULL,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                error = "timeout"
            except OSError as exc:
                error = f"os_error_{exc.__class__.__name__}"
            else:
                output = (result.stdout or "").strip()
                if result.returncode == 0 and output:
                    return output, None
                output = output or (result.stderr or "").strip()
                error = f"cli_exit_{result.returncode}"
            # Connection/transport failure: wait and retry.
            if attempt < self.connection_retries:
                time.sleep(self.connection_retry_wait_s)
        return output, error

    def _parse(self, raw: str) -> dict | None:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = _JSON_RE.search(text)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        try:
            belief = float(obj.get("belief_x_positive"))
            action = str(obj.get("action"))
            amount = float(obj.get("amount", 0.0))
        except (TypeError, ValueError):
            return None
        if action not in {"stake_yes", "stake_no", "pass"}:
            return None
        if not (0.0 <= belief <= 1.0) or not math.isfinite(amount):
            return None
        return {"belief_x_positive": belief, "action": action, "amount": amount}

    # ------------------------------------------------------- contribution

    def _to_contribution(
        self,
        parsed: dict,
        wealth: float,
        y: float,
        my_past: list[Contribution],
    ) -> Contribution | None:
        action = parsed["action"]
        amount = parsed["amount"]
        if action == "pass" or amount <= 0.0:
            return None

        existing_stake = sum(c.amount for c in my_past)
        entry_cost = 0.0 if my_past else self.phi * wealth * math.sqrt(y)
        # Worst-case loss must stay strictly below wealth (simulator enforces).
        budget = wealth - entry_cost - existing_stake * (1.0 + self.fee_rate)
        max_amount = 0.95 * budget / (1.0 + self.fee_rate)
        if max_amount <= 0.0:
            return None
        amount = min(amount, max_amount)
        if amount < 1e-6:
            return None

        side = "approve" if action == "stake_yes" else "reject"
        return Contribution(amount=float(amount), data={"side": side})

    # ------------------------------------------------------------- logging

    def _log(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
