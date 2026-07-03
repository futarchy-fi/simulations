"""Merge shard reports and compute all arm metrics for the LLM decision-market
experiment. Emits results/llm-decision-market/metrics.json and prints tables.

Arms:
  A  rational-heuristic market (bayesian_threshold + binary_staking_market)
  B  LLM market (haiku)
  C  poll control: mean of Arm B first-round (pre-history) beliefs > 0.5
  D  LLM manager with union of all signals (separate runner output)
  E  trivial baselines (random, always approve/reject, best-informed signal)
  B-sonnet  optional subsample
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parents[1] / "results" / "llm-decision-market"
LOGS = ROOT / "logs"


# ----------------------------------------------------------------- loading

def load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def merge_reports(paths: list[Path]) -> dict:
    rows = []
    durations = []
    for path in paths:
        report = load_report(path)
        rows.extend(report["per_proposal"])
        durations.append(report["metadata"]["duration_seconds"])
    rows.sort(key=lambda r: r["index"])
    return {"per_proposal": rows, "shard_durations": durations}


# ----------------------------------------------------------------- metrics

def decision_metrics(env_rows: list[dict], approve: dict[int, bool]) -> dict:
    n = len(env_rows)
    correct = 0
    value = 0.0
    value_star = 0.0
    for row in env_rows:
        d = bool(approve[row["index"]])
        good = row["x"] > 0.0
        if d == good:
            correct += 1
        if d:
            value += row["x"] * row["y"]
        if good:
            value_star += row["x"] * row["y"]
    return {
        "n": n,
        "accuracy": correct / n,
        "value": value,
        "value_star": value_star,
        "regret": value_star - value,
        "value_ratio": value / value_star if value_star else None,
    }


def market_extras(report_rows: list[dict]) -> dict:
    """Mechanism profit, oracle usage, participation, utilities."""
    profit = sum(r["mechanism_net_profit"] for r in report_rows)
    oracle_count = sum(1 for r in report_rows if r["use_futarchy"])
    per_agent: dict[str, dict] = {}
    participations = 0
    for row in report_rows:
        for agent in row["agent_reports"]:
            slot = per_agent.setdefault(
                agent["agent_instance_id"],
                {"wealth": agent["wealth"], "utility": 0.0, "stake": 0.0, "n_part": 0},
            )
            slot["utility"] += agent["utility"]
            slot["stake"] += agent["stake"]
            if agent["stake"] > 0:
                slot["n_part"] += 1
                participations += 1
    return {
        "mechanism_net_profit": profit,
        "oracle_invocations": oracle_count,
        "agent_participations": participations,
        "total_agent_utility": sum(a["utility"] for a in per_agent.values()),
        "per_agent": per_agent,
    }


# ----------------------------------------------------------- Arm B logs

def load_llm_logs(prefixes: tuple[str, ...], shard_offsets: dict[str, int]) -> list[dict]:
    records = []
    for path in sorted(glob.glob(str(LOGS / "calls_*.jsonl"))):
        name = Path(path).stem  # calls_<shard>_<uid>
        shard = name.split("_")[1]
        if not shard.startswith(prefixes):
            continue
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            rec["global_index"] = shard_offsets[shard] + rec["proposal_local_index"]
            records.append(rec)
    return records


def poll_decisions(records: list[dict], indices: list[int]) -> tuple[dict[int, bool], dict]:
    by_prop: dict[int, list[float]] = {}
    for rec in records:
        if rec["round_index"] != 0 or rec["parsed"] is None:
            continue
        by_prop.setdefault(rec["global_index"], []).append(
            rec["parsed"]["belief_x_positive"]
        )
    decisions = {}
    n_beliefs = []
    for idx in indices:
        beliefs = by_prop.get(idx, [])
        n_beliefs.append(len(beliefs))
        decisions[idx] = (sum(beliefs) / len(beliefs) > 0.5) if beliefs else False
    return decisions, {"mean_beliefs_per_proposal": float(np.mean(n_beliefs))}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def arm_b_microstructure(records: list[dict], env_rows: list[dict]) -> dict:
    env = {row["index"]: row for row in env_rows}

    # --- stake sizing vs informedness -------------------------------------
    # per-(agent,proposal) accepted stake vs signal error and precision
    stakes, abs_errors, precisions, extremity = [], [], [], []
    for rec in records:
        if rec["parsed"] is None or rec.get("contribution") is None:
            continue
        row = env.get(rec["global_index"])
        if row is None:
            continue
        stakes.append(rec["contribution"]["amount"])
        abs_errors.append(abs(rec["signal"] - row["x"]))
        precisions.append(2.0 * rec["wealth"])
        extremity.append(abs(rec["parsed"]["belief_x_positive"] - 0.5))
    stakes_arr = np.array(stakes)
    stake_stats = {
        "n_accepted_stakes": len(stakes),
        "spearman_stake_vs_precision": spearman(stakes_arr, np.array(precisions)),
        "spearman_stake_vs_abs_signal_error": spearman(stakes_arr, np.array(abs_errors)),
        "spearman_stake_vs_belief_extremity": spearman(stakes_arr, np.array(extremity)),
    }

    # per-agent aggregates (identical 5 wealth levels)
    per_agent: dict[str, dict] = {}
    for rec in records:
        slot = per_agent.setdefault(
            rec["instance_uid"],
            {"wealth": rec["wealth"], "stake": 0.0, "abs_err": [], "beliefs_correct": 0, "beliefs_n": 0},
        )
        row = env.get(rec["global_index"])
        if rec["round_index"] == 0 and row is not None:
            slot["abs_err"].append(abs(rec["signal"] - row["x"]))
            if rec["parsed"] is not None:
                slot["beliefs_n"] += 1
                if (rec["parsed"]["belief_x_positive"] > 0.5) == (row["x"] > 0):
                    slot["beliefs_correct"] += 1
        if rec.get("contribution"):
            slot["stake"] += rec["contribution"]["amount"]
    agent_rows = sorted(per_agent.values(), key=lambda s: s["wealth"])
    agent_table = [
        {
            "wealth": s["wealth"],
            "precision": 2.0 * s["wealth"],
            "mean_abs_signal_error": float(np.mean(s["abs_err"])) if s["abs_err"] else None,
            "total_stake": s["stake"],
            "belief_accuracy": s["beliefs_correct"] / s["beliefs_n"] if s["beliefs_n"] else None,
        }
        for s in agent_rows
    ]
    wealth_arr = np.array([s["wealth"] for s in agent_rows])
    stake_by_agent = np.array([s["stake"] for s in agent_rows])
    stake_stats["spearman_agent_wealth_vs_total_stake"] = spearman(wealth_arr, stake_by_agent)

    # --- within-market learning -------------------------------------------
    # final price vs first-round mean belief vs last-round mean belief
    first_beliefs: dict[int, list[float]] = {}
    for rec in records:
        if rec["parsed"] is None:
            continue
        idx = rec["global_index"]
        if rec["round_index"] == 0:
            first_beliefs.setdefault(idx, []).append(rec["parsed"]["belief_x_positive"])
    # last-round belief: highest round per (agent, proposal)
    best_round: dict[tuple[int, str], tuple[int, float]] = {}
    for rec in records:
        if rec["parsed"] is None:
            continue
        key = (rec["global_index"], rec["instance_uid"])
        prev = best_round.get(key)
        if prev is None or rec["round_index"] > prev[0]:
            best_round[key] = (rec["round_index"], rec["parsed"]["belief_x_positive"])

    last_by_prop: dict[int, list[float]] = {}
    for (idx, _uid), (_r, belief) in best_round.items():
        last_by_prop.setdefault(idx, []).append(belief)

    prices: dict[int, float] = {}
    for row in env_rows:
        yes = no = 0.0
        for agent in row["agent_reports"]:
            for attempt in agent["attempts"]:
                if attempt["accepted"] and attempt.get("contribution"):
                    side = attempt["contribution"]["data"]["side"]
                    amt = attempt["contribution"]["amount"]
                    if side == "approve":
                        yes += amt
                    else:
                        no += amt
        if yes + no > 0:
            prices[row["index"]] = yes / (yes + no)

    def predictor_stats(pred: dict[int, float]) -> dict:
        xs, ps = [], []
        for idx, p in pred.items():
            xs.append(env[idx]["x"])
            ps.append(p)
        xs_arr, ps_arr = np.array(xs), np.array(ps)
        acc = float(np.mean((ps_arr > 0.5) == (xs_arr > 0)))
        corr = float(np.corrcoef(ps_arr, xs_arr)[0, 1]) if len(xs) > 2 and ps_arr.std() > 0 else float("nan")
        return {"n": len(xs), "accuracy_sign": acc, "pearson_vs_x": corr}

    learning = {
        "first_round_mean_belief": predictor_stats(
            {i: float(np.mean(b)) for i, b in first_beliefs.items()}
        ),
        "last_round_mean_belief": predictor_stats(
            {i: float(np.mean(b)) for i, b in last_by_prop.items()}
        ),
        "final_market_price": predictor_stats(prices),
    }

    # cost accounting
    n_records = len(records)
    n_cli_calls = sum(r["raw_response"].count("---RETRY---") + 1 for r in records)
    latencies = [r["latency_s"] for r in records]
    cost = {
        "llm_act_calls": n_records,
        "cli_invocations": n_cli_calls,
        "parse_failures": sum(1 for r in records if r["error"] == "parse_failure"),
        "other_errors": sum(
            1 for r in records if r["error"] not in (None, "parse_failure")
        ),
        "mean_latency_s": float(np.mean(latencies)) if latencies else None,
        "p90_latency_s": float(np.percentile(latencies, 90)) if latencies else None,
    }

    return {"stake_sizing": stake_stats, "per_agent": agent_table, "learning": learning, "cost": cost}


# --------------------------------------------------------------------- main

def shard_offsets_for(total: int, num_shards: int, prefix: str) -> dict[str, int]:
    per = (total + num_shards - 1) // num_shards
    offsets = {}
    start = 0
    i = 0
    while start < total:
        offsets[f"{prefix}{i}"] = start
        start += min(per, total - start)
        i += 1
    return offsets


def main() -> None:
    out: dict = {}

    # Arm A -----------------------------------------------------------------
    arm_a = load_report(RESULTS / "arm_a_report.json")
    env_rows = arm_a["per_proposal"]
    a_dec = {r["index"]: r["final_decision"] == "approve" for r in env_rows}
    out["arm_A_rational_market"] = decision_metrics(env_rows, a_dec) | market_extras(env_rows)

    # Arm B -----------------------------------------------------------------
    b_paths = sorted(RESULTS.glob("arm_b_shard*_report.json"))
    b_offsets = shard_offsets_for(150, 8, "b")
    b_records = load_llm_logs(("b",), b_offsets)
    b_dec: dict[int, bool] | None = None
    c_dec: dict[int, bool] | None = None
    if b_paths:
        merged_b = merge_reports(b_paths)
        b_rows = merged_b["per_proposal"]
        # sanity: env identical to arm A
        for ra, rb in zip(env_rows, b_rows):
            assert abs(ra["x"] - rb["x"]) < 1e-12 and abs(ra["y"] - rb["y"]) < 1e-12, (
                f"env mismatch at {ra['index']}"
            )
        b_dec = {r["index"]: r["final_decision"] == "approve" for r in b_rows}
        out["arm_B_llm_market"] = (
            decision_metrics(env_rows, b_dec)
            | market_extras(b_rows)
            | {"shard_durations_s": merged_b["shard_durations"]}
        )
        out["arm_B_microstructure"] = arm_b_microstructure(b_records, b_rows)

        # Arm C ---------------------------------------------------------------
        indices = [r["index"] for r in env_rows]
        c_dec, c_meta = poll_decisions(b_records, indices)
        out["arm_C_poll_control"] = decision_metrics(env_rows, c_dec) | c_meta

    # Arm D -----------------------------------------------------------------
    d_path = RESULTS / "arm_d_decisions.json"
    if d_path.exists():
        arm_d = json.loads(d_path.read_text())
        d_dec = {r["index"]: r["decision"] == "approve" for r in arm_d["decisions"]}
        out["arm_D_llm_manager"] = decision_metrics(env_rows, d_dec) | {
            "total_calls": arm_d["total_calls"],
            "wall_time_s": arm_d["wall_time_s"],
            "failures": sum(1 for r in arm_d["decisions"] if r["failed"]),
        }

    # Arm E -----------------------------------------------------------------
    rng = np.random.default_rng(4242)
    rand_dec = {r["index"]: bool(rng.random() < 0.5) for r in env_rows}
    out["arm_E_random"] = decision_metrics(env_rows, rand_dec)
    out["arm_E_always_approve"] = decision_metrics(env_rows, {r["index"]: True for r in env_rows})
    out["arm_E_always_reject"] = decision_metrics(env_rows, {r["index"]: False for r in env_rows})
    # best-informed agent decides: sign of the max-precision agent's raw signal
    best_dec = {}
    for row in env_rows:
        best = max(row["agent_reports"], key=lambda a: a["wealth"])
        best_dec[row["index"]] = best["signal"] > 0
    out["arm_E_best_informed_signal"] = decision_metrics(env_rows, best_dec)

    # Sonnet subsample ------------------------------------------------------
    s_paths = sorted(RESULTS.glob("arm_b_sonnet_shard*_report.json"))
    if s_paths:
        merged_s = merge_reports(s_paths)
        s_rows = merged_s["per_proposal"]
        env_sub = env_rows[: len(s_rows)]
        s_dec = {r["index"]: r["final_decision"] == "approve" for r in s_rows}
        s_offsets = shard_offsets_for(30, 4, "sonnet")
        s_records = load_llm_logs(("sonnet",), s_offsets)
        out["arm_B_sonnet_subsample"] = (
            decision_metrics(env_sub, s_dec) | market_extras(s_rows)
        )
        out["arm_B_sonnet_microstructure"] = arm_b_microstructure(s_records, s_rows)
        # matched comparisons on the same 30 proposals
        out["matched_first30"] = {
            "arm_A": decision_metrics(env_sub, a_dec),
            "arm_B_haiku": decision_metrics(env_sub, b_dec) if b_dec is not None else None,
            "arm_C_poll": decision_metrics(env_sub, c_dec) if c_dec is not None else None,
        }
        s_indices = [r["index"] for r in s_rows]
        sc_dec, sc_meta = poll_decisions(s_records, s_indices)
        out["arm_C_sonnet_poll"] = decision_metrics(env_sub, sc_dec) | sc_meta

    # drop bulky per_agent dicts from headline metrics
    for key in ("arm_A_rational_market", "arm_B_llm_market", "arm_B_sonnet_subsample"):
        if key in out and "per_agent" in out[key]:
            out[key]["per_agent"] = {
                k: v for k, v in sorted(out[key]["per_agent"].items())
            }

    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float))

    # ------------------------------------------------------------- print
    print(f"{'arm':38s} {'n':>4s} {'acc':>6s} {'value':>9s} {'regret':>8s} {'ratio':>6s}")
    for key, val in out.items():
        if isinstance(val, dict) and "accuracy" in val:
            ratio = val.get("value_ratio")
            print(
                f"{key:38s} {val['n']:4d} {val['accuracy']:6.3f} {val['value']:9.2f} "
                f"{val['regret']:8.2f} {ratio if ratio is None else format(ratio, '6.3f')}"
            )
    print(f"\nwrote {RESULTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
