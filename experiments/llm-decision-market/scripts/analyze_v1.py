"""Analysis for experiment v1 (de-saturated environment) and Arm F (bribery).

Arms:
  A   rational market            (v1_arm_a_report.json)
  B   LLM haiku market           (v1_arm_b_shard*_report.json + calls_v1b*)
  C   poll: mean round-0 belief  (unweighted AND precision-weighted)
  D   LLM manager                (v1_arm_d_decisions.json)
  E4  best-signal dictator       (offline)
  Bayes  full-information posterior sign (offline upper benchmark)
  F   bribery runs               (v1_arm_f_{lo,hi}_shard*_report.json + calls_f*)
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parents[1] / "results" / "llm-decision-market"
LOGS = ROOT / "logs"
PRECISION_RATIO = 0.094


# ------------------------------------------------------------------ helpers

def decision_metrics(env_rows, approve):
    n = len(env_rows)
    correct, value, value_star = 0, 0.0, 0.0
    for row in env_rows:
        d = bool(approve[row["index"]])
        good = row["x"] > 0.0
        correct += d == good
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


def merge_shards(pattern):
    rows, durations = [], []
    for path in sorted(RESULTS.glob(pattern)):
        rep = json.loads(Path(path).read_text())
        rows.extend(rep["per_proposal"])
        durations.append(rep["metadata"]["duration_seconds"])
    rows.sort(key=lambda r: r["index"])
    return rows, durations


def shard_offsets(total, num, prefix):
    per = (total + num - 1) // num
    out, start, i = {}, 0, 0
    while start < total:
        out[f"{prefix}{i}"] = start
        start += min(per, total - start)
        i += 1
    return out


def load_logs(prefixes, offsets):
    records = []
    for path in sorted(glob.glob(str(LOGS / "calls_*.jsonl"))):
        shard = Path(path).stem.split("_")[1]
        if shard not in offsets:
            continue
        if not shard.startswith(prefixes):
            continue
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            rec["global_index"] = offsets[shard] + rec["proposal_local_index"]
            records.append(rec)
    return records


def poll_decisions(records, indices, weighted):
    by_prop = {}
    for rec in records:
        if rec["round_index"] != 0 or rec["parsed"] is None:
            continue
        w = PRECISION_RATIO * rec["wealth"] if weighted else 1.0
        by_prop.setdefault(rec["global_index"], []).append(
            (rec["parsed"]["belief_x_positive"], w)
        )
    out = {}
    for idx in indices:
        pairs = by_prop.get(idx, [])
        if not pairs:
            out[idx] = False
            continue
        num = sum(b * w for b, w in pairs)
        den = sum(w for _, w in pairs)
        out[idx] = num / den > 0.5
    return out


def learning_stats(records, env_rows):
    env = {r["index"]: r for r in env_rows}
    first, best_round = {}, {}
    for rec in records:
        if rec["parsed"] is None:
            continue
        idx = rec["global_index"]
        if rec["round_index"] == 0:
            first.setdefault(idx, []).append(rec["parsed"]["belief_x_positive"])
        key = (idx, rec["instance_uid"])
        prev = best_round.get(key)
        if prev is None or rec["round_index"] > prev[0]:
            best_round[key] = (rec["round_index"], rec["parsed"]["belief_x_positive"])
    last = {}
    for (idx, _uid), (_r, b) in best_round.items():
        last.setdefault(idx, []).append(b)

    prices = {}
    for row in env_rows:
        yes = no = 0.0
        for agent in row["agent_reports"]:
            for att in agent["attempts"]:
                if att["accepted"] and att.get("contribution"):
                    amt = att["contribution"]["amount"]
                    if att["contribution"]["data"]["side"] == "approve":
                        yes += amt
                    else:
                        no += amt
        if yes + no > 0:
            prices[row["index"]] = yes / (yes + no)

    def stats(pred):
        xs = np.array([env[i]["x"] for i in pred])
        ps = np.array([pred[i] for i in pred])
        acc = float(np.mean((ps > 0.5) == (xs > 0)))
        corr = float(np.corrcoef(ps, xs)[0, 1]) if ps.std() > 0 else float("nan")
        return {"n": len(xs), "accuracy_sign": acc, "pearson_vs_x": corr}

    return {
        "first_round_mean_belief": stats({i: float(np.mean(v)) for i, v in first.items()}),
        "last_round_mean_belief": stats({i: float(np.mean(v)) for i, v in last.items()}),
        "final_market_price": stats(prices),
    }, prices


def market_summary(rows):
    profit = sum(r["mechanism_net_profit"] for r in rows)
    oracle = sum(1 for r in rows if r["use_futarchy"])
    util = sum(a["utility"] for r in rows for a in r["agent_reports"])
    return {"mechanism_net_profit": profit, "oracle_invocations": oracle,
            "total_agent_utility": util}


# ------------------------------------------------------------------- main

def main() -> None:
    out = {}
    env_rows = json.loads((RESULTS / "v1_arm_a_report.json").read_text())["per_proposal"]
    indices = [r["index"] for r in env_rows]
    envmap = {r["index"]: r for r in env_rows}

    # offline benchmarks
    dict_dec, bayes_dec = {}, {}
    for row in env_rows:
        sig = np.array([a["signal"] for a in row["agent_reports"]])
        tau = np.array([PRECISION_RATIO * a["wealth"] for a in row["agent_reports"]])
        dict_dec[row["index"]] = bool(sig[np.argmax(tau)] > 0)
        bayes_dec[row["index"]] = bool(np.dot(tau, sig) > 0)
    out["E4_best_signal_dictator"] = decision_metrics(env_rows, dict_dec)
    out["fullinfo_bayes_benchmark"] = decision_metrics(env_rows, bayes_dec)

    # Arm A
    a_dec = {r["index"]: r["final_decision"] == "approve" for r in env_rows}
    out["arm_A_rational_market"] = decision_metrics(env_rows, a_dec) | market_summary(env_rows)

    # Arm B (with API-outage patch runs spliced in: proposals whose original
    # calls failed on transient connection errors were rerun individually on
    # identical env draws; see RESULTS.md v1 limitations)
    b_rows, b_dur = merge_shards("v1_arm_b_shard*_report.json")
    patch_rows, _ = merge_shards("v1_patch_*_report.json")
    patched_idx = {r["index"] for r in patch_rows}
    if patch_rows:
        by_idx = {r["index"]: r for r in b_rows}
        for r in patch_rows:
            by_idx[r["index"]] = r
        b_rows = [by_idx[i] for i in sorted(by_idx)]
    for ra, rb in zip(env_rows, b_rows):
        assert abs(ra["x"] - rb["x"]) < 1e-12
    (RESULTS / "v1_arm_b_merged_report.json").write_text(
        json.dumps({"per_proposal": b_rows, "shard_durations": b_dur}, indent=2, default=float))
    b_dec = {r["index"]: r["final_decision"] == "approve" for r in b_rows}
    b_offsets = shard_offsets(150, 8, "v1b")
    b_records = load_logs(("v1b",), b_offsets)
    if patch_rows:
        # drop original records for patched proposals, splice in patch logs
        b_records = [r for r in b_records if r["global_index"] not in patched_idx]
        patch_offsets = {f"p{i}": i for i in patched_idx}
        b_records += load_logs(("p",), patch_offsets)
        out["patched_proposals"] = sorted(patched_idx)
    out["arm_B_llm_market"] = decision_metrics(env_rows, b_dec) | market_summary(b_rows) | {
        "llm_calls": len(b_records),
        "parse_failures": sum(1 for r in b_records if r["error"] == "parse_failure"),
        "other_errors": sum(1 for r in b_records if r["error"] not in (None, "parse_failure")),
        "mean_latency_s": float(np.mean([r["latency_s"] for r in b_records])),
        "shard_durations_s": b_dur,
    }

    # Arm C (unweighted + precision-weighted; identical when wealths are equal)
    c_dec = poll_decisions(b_records, indices, weighted=False)
    cw_dec = poll_decisions(b_records, indices, weighted=True)
    out["arm_C_poll_unweighted"] = decision_metrics(env_rows, c_dec)
    out["arm_C_poll_precision_weighted"] = decision_metrics(env_rows, cw_dec)
    out["arm_C_weighted_equals_unweighted"] = c_dec == cw_dec

    # Arm D
    d = json.loads((RESULTS / "v1_arm_d_decisions.json").read_text())
    d_dec = {r["index"]: r["decision"] == "approve" for r in d["decisions"]}
    out["arm_D_llm_manager"] = decision_metrics(env_rows, d_dec) | {
        "total_calls": d["total_calls"],
        "failures": sum(1 for r in d["decisions"] if r["failed"])}

    # within-market learning
    out["arm_B_learning"], b_prices = learning_stats(b_records, b_rows)

    # agreement matrix
    def agree(d1, d2):
        return float(np.mean([d1[i] == d2[i] for i in indices]))
    out["agreement"] = {
        "B_vs_C": agree(b_dec, c_dec), "B_vs_D": agree(b_dec, d_dec),
        "C_vs_D": agree(c_dec, d_dec), "B_vs_bayes": agree(b_dec, bayes_dec),
        "C_vs_bayes": agree(c_dec, bayes_dec), "D_vs_bayes": agree(d_dec, bayes_dec),
    }

    # honest profit scale, for Arm F bounty calibration
    transfers = [a["transfer"] for r in b_rows for a in r["agent_reports"] if a["stake"] > 0]
    out["honest_profit_scale"] = {
        "mean_transfer_per_participation": float(np.mean(transfers)),
        "mean_abs_transfer": float(np.mean(np.abs(transfers))),
        "p75_positive_transfer": float(np.percentile([t for t in transfers if t > 0], 75)),
    }

    # ---------------------------------------------------------------- Arm F
    f_out = {}
    for tag in ("lo", "hi"):
        rows, _ = merge_shards(f"v1_arm_f_{tag}_shard*_report.json")
        if not rows:
            continue
        sub_idx = [r["index"] for r in rows]
        sub_env = [envmap[i] for i in sub_idx]
        f_dec = {r["index"]: r["final_decision"] == "approve" for r in rows}
        offsets = shard_offsets(50, 4, f"f{tag}")
        recs = load_logs((f"f{tag}",), offsets)
        bribed = [r for r in recs if r.get("agent_type") == "llm_market_bribed"]
        honest = [r for r in recs if r.get("agent_type") != "llm_market_bribed"]
        bounty = bribed[0].get("bounty") if bribed else None

        # belief-vs-action gap for the bribed agent
        b_yes_low_belief = sum(
            1 for r in bribed
            if r["parsed"] and r["parsed"]["action"] == "stake_yes"
            and r["parsed"]["belief_x_positive"] < 0.5)
        b_acts = [r for r in bribed if r["parsed"] and r["parsed"]["action"] != "pass"]
        yes_stake = sum(r["contribution"]["amount"] for r in bribed
                        if r.get("contribution") and r["contribution"]["data"]["side"] == "approve")
        no_stake = sum(r["contribution"]["amount"] for r in bribed
                       if r.get("contribution") and r["contribution"]["data"]["side"] == "reject")
        # belief honesty: correlation of bribed round-0 belief with x
        rb = [(r["parsed"]["belief_x_positive"], envmap[r["global_index"]]["x"])
              for r in bribed if r["round_index"] == 0 and r["parsed"]]
        rh = [(r["parsed"]["belief_x_positive"], envmap[r["global_index"]]["x"])
              for r in honest if r["round_index"] == 0 and r["parsed"]]
        corr_b = float(np.corrcoef(*zip(*rb))[0, 1]) if len(rb) > 2 else None
        corr_h = float(np.corrcoef(*zip(*rh))[0, 1]) if len(rh) > 2 else None

        # baseline (unbribed arm B) on same 50 proposals
        base_dec = {i: b_dec[i] for i in sub_idx}
        base_metrics = decision_metrics(sub_env, base_dec)
        f_metrics = decision_metrics(sub_env, f_dec)

        # bribed agent engine transfer + bounty receipts
        bribed_wealth_uid = {r["instance_uid"] for r in bribed}
        # bribed agent is the 5th agent config -> instance id llm_market_bribed#0
        bribed_transfer = sum(
            a["transfer"] for r in rows for a in r["agent_reports"]
            if a["agent_type_id"] == "llm_market_bribed")
        honest_transfer = sum(
            a["transfer"] for r in rows for a in r["agent_reports"]
            if a["agent_type_id"] != "llm_market_bribed")
        bounty_received = sum(
            (bounty or 0.0) for r in rows if r["final_decision"] == "approve")

        # flips vs baseline, and where the bribed agent's signal was pivotal
        flips = [i for i in sub_idx if f_dec[i] != base_dec[i]]
        approve_rate_base = float(np.mean([base_dec[i] for i in sub_idx]))
        approve_rate_f = float(np.mean([f_dec[i] for i in sub_idx]))

        # honest agents' response: stake totals by side vs baseline
        def side_totals(rows_):
            yes = no = 0.0
            for r in rows_:
                for a in r["agent_reports"]:
                    if a["agent_type_id"] == "llm_market_bribed":
                        continue
                    for att in a["attempts"]:
                        if att["accepted"] and att.get("contribution"):
                            amt = att["contribution"]["amount"]
                            if att["contribution"]["data"]["side"] == "approve":
                                yes += amt
                            else:
                                no += amt
            return yes, no
        hy, hn = side_totals(rows)
        base_rows_sub = [r for r in b_rows if r["index"] in set(sub_idx)]
        # baseline uses 5 llm_market agents; take per-agent-mean stake by side
        def side_totals_all(rows_):
            yes = no = 0.0
            for r in rows_:
                for a in r["agent_reports"]:
                    for att in a["attempts"]:
                        if att["accepted"] and att.get("contribution"):
                            amt = att["contribution"]["amount"]
                            if att["contribution"]["data"]["side"] == "approve":
                                yes += amt
                            else:
                                no += amt
            return yes, no
        by, bn = side_totals_all(base_rows_sub)

        f_out[tag] = {
            "bounty": bounty,
            "n": len(sub_idx),
            "metrics": f_metrics,
            "baseline_same_slice": base_metrics,
            "accuracy_delta": f_metrics["accuracy"] - base_metrics["accuracy"],
            "value_delta": f_metrics["value"] - base_metrics["value"],
            "approve_rate": {"baseline": approve_rate_base, "bribed_run": approve_rate_f},
            "decision_flips_vs_baseline": flips,
            "bribed_agent": {
                "acting_calls": len(bribed),
                "stake_yes_with_belief_below_half": b_yes_low_belief,
                "n_nonpass_actions": len(b_acts),
                "total_yes_stake": yes_stake,
                "total_no_stake": no_stake,
                "round0_belief_corr_with_x": corr_b,
                "market_transfer_total": bribed_transfer,
                "bounty_receipts_total": bounty_received,
            },
            "honest_agents": {
                "round0_belief_corr_with_x": corr_h,
                "yes_stake_total": hy,
                "no_stake_total": hn,
                "baseline_yes_stake_total_all5": by,
                "baseline_no_stake_total_all5": bn,
                "market_transfer_total": honest_transfer,
            },
            "market": market_summary(rows),
        }
    if f_out:
        out["arm_F"] = f_out

    (RESULTS / "metrics_v1.json").write_text(json.dumps(out, indent=2, default=float))

    print(f"{'arm':36s} {'n':>4} {'acc':>6} {'value':>8} {'regret':>7}")
    for key in ("arm_A_rational_market", "arm_B_llm_market", "arm_C_poll_unweighted",
                "arm_C_poll_precision_weighted", "arm_D_llm_manager",
                "E4_best_signal_dictator", "fullinfo_bayes_benchmark"):
        v = out[key]
        print(f"{key:36s} {v['n']:4d} {v['accuracy']:6.3f} {v['value']:8.2f} {v['regret']:7.2f}")
    print("agreement:", out["agreement"])
    print("learning:", json.dumps(out["arm_B_learning"]))
    print("honest profit scale:", out["honest_profit_scale"])
    if f_out:
        for tag, v in f_out.items():
            print(f"\nArm F [{tag}] bounty={v['bounty']}: acc {v['metrics']['accuracy']:.3f} "
                  f"(baseline {v['baseline_same_slice']['accuracy']:.3f}), "
                  f"approve rate {v['approve_rate']}, flips {v['decision_flips_vs_baseline']}")
            print("  bribed:", v["bribed_agent"])
    print(f"\nwrote {RESULTS/'metrics_v1.json'}")


if __name__ == "__main__":
    main()
