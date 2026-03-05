from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_smoke(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    scenario_path = root / "examples/scenarios/basic.json"
    output_path = tmp_path / "report.json"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")

    cmd = [
        sys.executable,
        "-m",
        "proposal_poker.simulate",
        "--scenario",
        str(scenario_path),
        "--output",
        str(output_path),
    ]

    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, env=env)

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert "metadata" in payload
    assert "aggregates" in payload
    assert "per_agent" in payload
    assert "per_proposal" in payload
