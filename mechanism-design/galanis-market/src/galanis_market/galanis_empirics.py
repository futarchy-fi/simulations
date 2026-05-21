"""Empirical numbers from Galanis (2026) for direct comparison.

Sources within the paper (arXiv:2604.20050v2):

* Page 2 abstract:
    - median market across all structures: price 0.91 when X = 1
    - Easy and Medium: "almost 1"
    - Hard: 0.73
    - Very Hard: 0.50  (coin toss)
* Section 6.1, Table 6, median (tau = 0.5) regression:
    - Easy baseline intercept ≈ 0.018  (-> implied price ≈ 98%)
    - Medium "performs identically well" (deviation ≈ 0)
    - Hard:        +0.286 over Easy baseline -> log err ≈ 0.304 -> price ≈ 0.74
    - Very Hard:   +0.700 over Easy baseline -> log err ≈ 0.718 -> price ≈ 0.49
* Table 5 (Information Aggregation, Both Waves, rounds=3):
    - Easy mean log err: 0.131
    - Medium mean log err: 0.173
    - Hard mean log err: 0.407
    - Very Hard mean log err: 0.469

We expose these as a dict so downstream comparisons can pull them by name.
"""

from __future__ import annotations

from typing import Dict


# Implied "typical" final price under each structure, reading off the
# Galanis quantile (median) regression.
EMPIRICAL_MEDIAN_PRICE_AT_X1: Dict[str, float] = {
    "t3s111y2": 0.98,    # Easy
    "t3s110": 0.98,      # Medium
    "t3s111": 0.75,      # Hard
    "t3s111o2ye2": 0.50, # Very Hard
}

# Mean log error from Both Waves (Table 5), 3-round configuration.
EMPIRICAL_MEAN_LOG_ERROR_3R: Dict[str, float] = {
    "t3s111y2": 0.131,
    "t3s110": 0.173,
    "t3s111": 0.407,
    "t3s111o2ye2": 0.469,
}

# Median log error implied by the quantile regression (Table 6, column 2).
# Easy intercept = 0.018; Medium ~ same; Hard = 0.018 + 0.286 = 0.304;
# Very Hard = 0.018 + 0.700 = 0.718.
EMPIRICAL_MEDIAN_LOG_ERROR: Dict[str, float] = {
    "t3s111y2": 0.018,
    "t3s110": 0.018,
    "t3s111": 0.304,
    "t3s111o2ye2": 0.718,
}


__all__ = [
    "EMPIRICAL_MEDIAN_PRICE_AT_X1",
    "EMPIRICAL_MEAN_LOG_ERROR_3R",
    "EMPIRICAL_MEDIAN_LOG_ERROR",
]
