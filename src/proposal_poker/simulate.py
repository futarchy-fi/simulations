"""CLI entrypoint for running Proposal Poker simulations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .discovery import discover_submissions
from .scenario import load_scenario
from .simulator import run_simulation


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Proposal Poker simulations")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON file")
    parser.add_argument(
        "--extensions-dir",
        action="append",
        default=[],
        help="Optional extra directory containing agents/ and mechanisms/ folders",
    )
    parser.add_argument("--output", help="Optional path to write JSON report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    config = load_scenario(args.scenario)

    repo_root = Path(__file__).resolve().parents[2]
    built_in_dir = repo_root / "mechanism-design/proposal-evaluation"

    registry = discover_submissions(
        repo_dirs=[built_in_dir],
        extension_dirs=args.extensions_dir,
    )
    report = run_simulation(config, registry=registry)

    payload = report.model_dump_json(indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
