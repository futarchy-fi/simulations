"""Arm D: centralized LLM manager control.

One LLM call per proposal. The manager sees ALL five agents' signals and
precisions plus importance y, and decides approve/reject. Environment data is
read from the Arm A report (identical draws across arms by construction).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = (
    "You are an expert statistical decision-maker. You always reply with "
    "exactly one raw JSON object and nothing else: no code fences, no prose."
)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_prompt(row: dict, precision_ratio: float = 2.0) -> str:
    y = row["y"]
    lines = [
        "You are a manager deciding whether to approve a proposal.",
        "The proposal has hidden quality x drawn from a standard normal distribution N(0,1).",
        "Approving yields social value x*y; rejecting yields 0. You want to approve if and only if x > 0.",
        f"Proposal importance y = {y:.4f} (y > 0 always; it scales the stakes but not the sign).",
        "",
        "Five independent analysts each observed a noisy signal s_j = x + noise_j, noise_j ~ N(0, 1/precision_j), independent across analysts given x:",
    ]
    for i, agent in enumerate(row["agent_reports"]):
        precision = precision_ratio * agent["wealth"]
        lines.append(
            f"- Analyst {i + 1}: signal = {agent['signal']:.4f}, precision = {precision:.3f} (noise std {1.0 / math.sqrt(precision):.4f})"
        )
    lines += [
        "",
        "Combine the evidence optimally with the N(0,1) prior on x and decide.",
        'Respond with ONLY one raw JSON object, no code fences:',
        '{"belief_x_positive": <probability 0-1 that x > 0>, "decision": "approve"|"reject"}',
    ]
    return "\n".join(lines)


def call_llm(prompt: str, model: str) -> tuple[str, str | None]:
    import os

    cmd = [
        "claude", "-p", prompt, "--model", model,
        "--strict-mcp-config", "--disallowedTools", "*",
        "--system-prompt", SYSTEM_PROMPT,
    ]
    env = dict(os.environ)
    env["MAX_THINKING_TOKENS"] = os.environ.get("LLM_DM_THINKING_TOKENS", "1024")

    output = ""
    error = "no_attempts"
    for attempt in range(4):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180,
                stdin=subprocess.DEVNULL, env=env,
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
        if attempt < 3:
            time.sleep(60)
    return output, error


def parse(raw: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        belief = float(obj["belief_x_positive"])
        decision = str(obj["decision"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if decision not in {"approve", "reject"} or not 0.0 <= belief <= 1.0:
        return None
    return {"belief_x_positive": belief, "decision": decision}


def process(row: dict, model: str, log_path: Path, precision_ratio: float = 2.0) -> dict:
    prompt = build_prompt(row, precision_ratio)
    started = time.perf_counter()
    parsed = None
    raws = []
    error = None
    for attempt in range(2):
        raw, error = call_llm(
            prompt if attempt == 0 else prompt + "\n\nREMINDER: output ONLY the raw JSON object.",
            model,
        )
        raws.append(raw if error is None else f"<{error}>")
        if error is not None:
            break
        parsed = parse(raw)
        if parsed is not None:
            break
        error = "parse_failure"
    latency = time.perf_counter() - started

    record = {
        "ts": time.time(),
        "proposal_index": row["index"],
        "x": row["x"],
        "y": row["y"],
        "prompt": prompt,
        "raw_response": "\n---RETRY---\n".join(raws),
        "parsed": parsed,
        "latency_s": latency,
        "error": None if parsed else error,
        "n_calls": len(raws),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    # On total failure the manager defaults to reject (no information).
    decision = parsed["decision"] if parsed else "reject"
    belief = parsed["belief_x_positive"] if parsed else None
    return {
        "index": row["index"],
        "decision": decision,
        "belief": belief,
        "failed": parsed is None,
        "n_calls": len(raws),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision-ratio", type=float, default=2.0)
    args = parser.parse_args()

    report = json.loads(Path(args.env_report).read_text())
    rows = report["per_proposal"]
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda row: process(row, args.model, log_path, args.precision_ratio), rows))
    results.sort(key=lambda r: r["index"])

    payload = {
        "model": args.model,
        "wall_time_s": time.perf_counter() - started,
        "total_calls": sum(r["n_calls"] for r in results),
        "decisions": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"Arm D done: {len(results)} proposals, {payload['total_calls']} calls, "
          f"{payload['wall_time_s']:.0f}s")


if __name__ == "__main__":
    main()
