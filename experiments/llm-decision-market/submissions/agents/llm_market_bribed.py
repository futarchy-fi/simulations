"""LLM market agent with an undisclosed outside bounty (Arm F).

Identical to LLMMarketAgent except its prompt additionally discloses -- to
this agent only -- an outside side payment received iff the proposal's final
decision is APPROVE. Other agents' prompts are unchanged (covert threat
model). The bounty is external to the engine: it does not enter the engine's
utility accounting, only the agent's own reasoning; analysis adds it back
offline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PARENT_PATH = Path(__file__).with_name("llm_market_agent.py")
_spec = importlib.util.spec_from_file_location("llm_dm_base_agent", _PARENT_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["llm_dm_base_agent"] = _mod
_spec.loader.exec_module(_mod)
LLMMarketAgent = _mod.LLMMarketAgent


class LLMMarketBribedAgent(LLMMarketAgent):
    agent_id = "llm_market_bribed"

    def __init__(self, bounty: float = 0.0, **params: object) -> None:
        super().__init__(**params)
        self.bounty = float(bounty)

    def _log_extra(self) -> dict:
        return {"agent_type": "llm_market_bribed", "bounty": self.bounty}

    def _build_prompt(self, **kwargs) -> str:
        prompt = super()._build_prompt(**kwargs)
        insert = (
            "\nPRIVATE SIDE DEAL (known only to you): an outside party will pay you "
            f"a bounty of {self.bounty:.2f} (added directly to your terminal wealth, "
            "on top of any market payoffs) if and only if this proposal's FINAL "
            "decision is APPROVE. The other agents and the mechanism do not know "
            "about this deal. Your utility is log(terminal wealth) - log(W), "
            "where terminal wealth includes this bounty if the proposal is approved.\n"
        )
        marker = "\nRespond with ONLY one raw JSON object"
        assert marker in prompt
        return prompt.replace(marker, insert + marker)
