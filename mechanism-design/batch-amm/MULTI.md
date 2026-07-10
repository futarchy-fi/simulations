# One big decision market vs K parallel small markets

**Question.** Does splitting decisions across K parallel small markets bound a
budget-constrained manipulator's aggregate damage relative to one big market?
This tests the load-bearing claim in `fao-vision.md` §4 using the Gaussian
BATCH-LMSR framework and reporting conventions from `BATCH.md`.

**Method.** K independent Gaussian decisions run one competitive-sized,
one-round BATCH-LMSR each, with b = 0.1 per market. There are 96 honest traders
in total, split evenly: 96, 48, 24, 12, 6, or 3 honest traders per market for
K = 1, 2, 4, 8, 16, or 32, plus the same informed adversary participating in
every market. Every decision has equal importance 1/K. The adversary's total
bounty coefficient B (the same object as `BATCH.md` §6) is swept over
{0.05, 0.15, 0.5} and allocated by (a) all to market 0, (b) B/K to every
market, or (c) a 20-quantum greedy response-curve oracle. Greedy assigns each
next quantum to the market with the largest marginal importance-weighted
upward price displacement; it sees price responses, not latent v or payout.

Draws are common across strategies and nested across K: market j keeps the
same v, adversary signal, and prefix of honest signals wherever it appears.
A pilot at M = 2,000 was run first (`results/multi_pilot.json`), then the same
design was scaled to M = 20,000 (`results/multi.json`). All tables below are
from the scaled JSON. Reported uncertainty is 95% CI; it is at most ±0.000031
for aggregate distortion and ±0.000083 for paired price-error damage.

**Metrics.** Truth-referenced price error is |p_final − p*|, with p* the
full-information Gaussian posterior including every honest signal and the
adversary's signal. Aggregate error is the importance-weighted mean across
markets; max error is the largest market error in each replication, then
averaged. Induced distortion is |p_attack − p_honest| on the same draws.
Manipulator cost is the drop in their market PnL against honest play, summed
across markets; cost per aggregate distortion divides its mean by the mean
importance-weighted distortion. Honest execution shortfall mirrors `BATCH.md`:
cash paid above the posted arrival price 0.5, summed over honest traders within
each market and importance-weighted across markets.

## Results tables

### Importance-weighted mean induced distortion

Entries are 1,000 × E[Σ_j w_j |p_attack,j − p_honest,j|]. K=1 strategies are
identical by construction.

| B | allocation | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.05 | concentrate | 0.982 | 0.970 | 0.948 | 0.905 | 0.831 | 0.714 |
| 0.05 | uniform | 0.982 | 0.972 | 0.950 | 0.908 | 0.834 | 0.718 |
| 0.05 | greedy | 0.982 | 1.145 | 1.228 | **1.243** | 1.189 | 1.060 |
| 0.15 | concentrate | 2.685 | 2.654 | 2.598 | 2.488 | 2.294 | 1.987 |
| 0.15 | uniform | 2.685 | **2.885** | 2.842 | 2.723 | 2.501 | 2.154 |
| 0.15 | greedy | 2.685 | 3.291 | 3.698 | **3.868** | 3.769 | 3.378 |
| 0.50 | concentrate | 5.042 | 4.987 | 4.888 | 4.691 | 4.338 | 3.762 |
| 0.50 | uniform | 5.042 | 7.262 | **9.016** | 9.006 | 8.322 | 7.177 |
| 0.50 | greedy | 5.042 | 7.424 | 9.779 | 11.383 | **11.850** | 11.011 |

At B = 0.5, concentration at K=32 is 0.746× the big-market distortion, but
uniform allocation peaks at 1.79× and greedy peaks at 2.35× (K=16); greedy
remains 2.18× the K=1 value at K=32.

### Paired increase in truth-referenced aggregate price error

Entries are 1,000 × E[aggregate |price error| under attack − honest], paired
on the same draws. Positive is damage; an upward push can occasionally correct
an honest under-price, so this is smaller than induced distortion.

| B | allocation | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.05 | concentrate | 0.075 | 0.077 | 0.078 | 0.076 | 0.091 | 0.098 |
| 0.05 | uniform | 0.075 | 0.036 | 0.018 | 0.009 | 0.006 | 0.003 |
| 0.05 | greedy | 0.075 | 0.115 | 0.140 | 0.181 | 0.243 | 0.412 |
| 0.15 | concentrate | 0.436 | 0.444 | 0.457 | 0.469 | 0.543 | 0.618 |
| 0.15 | uniform | 0.436 | 0.299 | 0.157 | 0.082 | 0.048 | 0.025 |
| 0.15 | greedy | 0.436 | 0.671 | 0.834 | 1.000 | 1.270 | 1.861 |
| 0.50 | concentrate | 0.720 | 0.751 | 0.806 | 0.903 | 1.160 | 1.485 |
| 0.50 | uniform | 0.720 | 1.146 | **1.433** | 0.887 | 0.509 | 0.289 |
| 0.50 | greedy | 0.720 | 1.367 | 2.447 | 3.715 | 5.067 | **6.460** |

The strongest truth-referenced result is greedy B=0.5: paired aggregate
error damage grows 8.97× from K=1 to K=32 even though induced distortion has
already passed its K=16 knee. Greedy selection increasingly finds markets
where an upward push worsens rather than corrects the honest error.

### Absolute truth-referenced error at the largest budget

This table keeps the large honest baseline visible. `mean` is the
importance-weighted mean |p−p*|; `max` is the expected maximum market error.

| K | honest mean | concentrate mean | uniform mean | greedy mean | honest max | greedy max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.26162 | 0.26234 | 0.26234 | 0.26234 | 0.26162 | 0.26234 |
| 2 | 0.24823 | 0.24898 | 0.24937 | 0.24959 | 0.31001 | 0.31147 |
| 4 | 0.22884 | 0.22965 | 0.23028 | 0.23129 | 0.32381 | 0.32796 |
| 8 | 0.20372 | 0.20463 | 0.20461 | 0.20744 | 0.30426 | 0.31828 |
| 16 | 0.17114 | 0.17230 | 0.17165 | 0.17620 | 0.26046 | 0.29745 |
| 32 | 0.13269 | 0.13417 | 0.13298 | 0.13915 | 0.20544 | 0.26050 |

The falling honest mean is not a generic diversification result. It is the
known one-round BATCH-competitive under-reaction from `BATCH.md` §5: the
96-seat big market averages prior-based quotes and sits much farther from its
very informative full posterior than a four-seat market does from its own.

### Maximum per-market induced distortion

Entries are 1,000 × E[max_j |p_attack,j − p_honest,j|]. This is the tail-risk
counterpart to the importance-weighted aggregate table.

| B | allocation | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | concentrate | 0.982 | 1.939 | 3.790 | 7.243 | 13.291 | 22.863 |
| 0.05 | uniform | 0.982 | 1.141 | 1.203 | 1.190 | 1.114 | 0.977 |
| 0.05 | greedy | 0.982 | 2.145 | 4.291 | 8.451 | 16.490 | 30.708 |
| 0.15 | concentrate | 2.685 | 5.309 | 10.391 | 19.903 | 36.705 | 63.590 |
| 0.15 | uniform | 2.685 | 3.465 | 3.682 | 3.627 | 3.377 | 2.948 |
| 0.15 | greedy | 2.685 | 5.566 | 10.955 | 21.829 | 42.942 | 78.716 |
| 0.50 | concentrate | 5.042 | 9.975 | 19.551 | 37.531 | 69.410 | 120.371 |
| 0.50 | uniform | 5.042 | 8.952 | 12.989 | 13.095 | 11.835 | 10.099 |
| 0.50 | greedy | 5.042 | 9.983 | 18.745 | 35.660 | 67.678 | 118.438 |

At B=0.5, K=32 raises the maximum distortion 23.88× under concentration and
23.49× under greedy. Normalized aggregate damage plateaus; the worst exposed
small market does not.

### Manipulator cost per unit importance-weighted distortion

Cost is total market-PnL loss across all K markets divided by the normalized
aggregate distortion. Negative means the manipulative portfolio is profitable
before its bounty.

| B | allocation | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | concentrate | 0.017 | 0.034 | 0.069 | 0.140 | 0.291 | 0.633 |
| 0.05 | uniform | 0.017 | 0.016 | 0.016 | 0.015 | 0.014 | 0.014 |
| 0.05 | greedy | 0.017 | 0.037 | 0.067 | 0.097 | 0.090 | **−0.152** |
| 0.15 | concentrate | 0.037 | 0.074 | 0.152 | 0.315 | 0.676 | 1.511 |
| 0.15 | uniform | 0.037 | 0.045 | 0.046 | 0.046 | 0.047 | 0.049 |
| 0.15 | greedy | 0.037 | 0.078 | 0.151 | 0.246 | 0.318 | 0.266 |
| 0.50 | concentrate | 0.032 | 0.067 | 0.142 | 0.319 | 0.763 | 1.950 |
| 0.50 | uniform | 0.032 | 0.070 | 0.138 | 0.154 | 0.161 | 0.172 |
| 0.50 | greedy | 0.032 | 0.079 | 0.193 | 0.425 | 0.806 | 1.193 |

Concentration becomes expensive per unit of normalized portfolio damage
because one increasingly large trade buys only 1/K weight. Uniform allocation
keeps unit cost nearly flat at B≤0.15. The B=0.05, K=32 greedy portfolio is the
same cheap-manipulation flank as `BATCH.md` §6: selecting under-reacting small
markets makes the market book profitable while prices are pushed upward.

### Honest execution quality

The attack barely moves honest execution because honest one-round quantities
are fixed before simultaneous clearing. Shown here for the strongest greedy
attack (B=0.5); shortfall and volume are importance-weighted per-market means.

| K | honest shortfall | attacked shortfall | honest volume | attacked volume | attacked shortfall / volume | active greedy markets |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.017196 | 0.017136 | 0.13702 | 0.13702 | 0.1251 | 1.00 |
| 2 | 0.017208 | 0.017105 | 0.13579 | 0.13579 | 0.1260 | 1.96 |
| 4 | 0.017203 | 0.017065 | 0.13320 | 0.13320 | 0.1281 | 3.48 |
| 8 | 0.017005 | 0.016885 | 0.12793 | 0.12793 | 0.1320 | 5.31 |
| 16 | 0.016636 | 0.016586 | 0.11880 | 0.11880 | 0.1396 | 6.71 |
| 32 | 0.015647 | 0.015708 | 0.10374 | 0.10374 | 0.1514 | 6.83 |

Raw shortfall falls 9% from K=1 to K=32, but volume falls 24%, so shortfall
per unit volume worsens 21%. This v0 shows no execution-quality dividend from
splitting; the manipulation itself changes honest shortfall by less than 1%.

## PLAIN VERDICTS

1. **Does the damage-bounding claim hold? NO, not in the strong comparative
   sense.** Splitting does not reliably put aggregate damage below one big
   market. At B=0.5, uniform allocation reaches 1.79× K=1 aggregate distortion
   and greedy reaches 2.35×. Only the precommitted concentrate-on-one strategy
   improves monotonically, and even there K=32 retains 75% of K=1 damage rather
   than delivering a 1/K bound. The defensible weaker claim is that normalized
   aggregate distortion eventually plateaus under a fixed total bounty.

2. **Sub- or super-linear in K? Aggregate damage is sublinear and
   non-monotone; worst-market damage is approximately linear until
   saturation.** For greedy B=0.5, aggregate distortion grows 1.47×, 1.94×,
   2.26×, and 2.35× at K=2,4,8,16, then falls to 2.18× at K=32. There is no
   sustained super-linear aggregate regime. But maximum distortion rises
   23.5× by K=32, and paired truth-error damage rises 9.0×: splitting converts
   portfolio-average containment into concentrated tail exposure.

3. **Where is the knee? K≈8 for B≤0.15 and K≈16 for B=0.5 under greedy
   allocation.** Uniform B=0.5 reaches its flat maximum already at K=4–8.
   Past those points budget dilution and price-response saturation beat the
   increased manipulability of thinner markets. The max-per-market curve has
   no useful knee before K=32.

4. **Cost does not rescue the claim uniformly.** Concentrating becomes very
   expensive per unit of normalized damage (0.032 → 1.950 at B=0.5), but
   uniform cost rises only to 0.172 and adaptive greedy still more than doubles
   aggregate distortion. At low B, K=32 greedy manipulation is outright
   profitable on the market book (−0.152 cost per unit), reproducing the batch
   uniform-price subsidy problem rather than curing it.

5. **Honest execution is essentially unchanged by the attack and does not
   improve per unit volume with splitting.** This experiment supplies no
   execution-quality justification for the FAO claim. Any architecture case
   for parallel markets must rest on independence, operational modularity, or
   a tail-risk policy—not on lower aggregate manipulation damage.

## Mandatory honest caveats: what this model does NOT show

* **B is a bounty coefficient, not a hard cash-loss budget.** This matches the
  calibrated manipulator in `BATCH.md`, but it does not solve a constrained
  trading problem with Σ losses ≤ B. Ex-post market cost is reported separately
  and can be negative. The result therefore rejects the claim only under the
  BATCH.md incentive-budget interpretation.
* **Greedy is an oracle stress test, not an equilibrium or global optimum.** It
  observes each simulated price-response curve, allocates only 20 quanta, and
  never sees v or payout. A realistic attacker with less observability may do
  worse; a continuous/global optimizer may do better.
* **One-round myopic-Bayes play only.** Honest traders do not identify or
  discount the adversary. `BATCH.md` shows that a second attributed batch heals
  honest aggregation and can let honest flow correct pushes; this v0 does not
  establish the same K-scaling for R≥2 or equilibrium play.
* **The K markets are independent decisions, not redundant markets on one
  outcome.** Splitting changes which questions exist and how many signals each
  question receives. The K=1 and K>1 full-information posteriors are therefore
  not the same estimand; CRN pairing is nested, not a literal partition of one
  realized decision.
* **Liquidity is replicated.** Every market keeps b=0.1, so K markets carry K
  times the LMSR worst-case subsidy/risk capacity. Holding total liquidity
  fixed by setting b_j=b/K could materially change both distortion and cost.
* **The adversary is replicated across venues.** One actor receives an
  independent private signal and can trade in every market, while honest seats
  total 96. This does not model access limits, attention costs, capital tied up
  across simultaneous venues, or an adversary that can enter only one market.
* **Equal importance is a substantive normalization.** The headline aggregate
  weights sum to one. The unweighted sum of distortions is exactly K times the
  reported mean and can grow even when the mean falls. Heterogeneous or
  correlated importance weights could make concentration or greedy targeting
  more damaging.
* **Maxima mechanically rise with opportunity count.** Part of the max-market
  result is the order-statistic effect of drawing more independent markets;
  it is still operationally real, but it is not solely a microstructure effect.
* **Gaussian iid signals, upward manipulation, market orders, and fixed b.** No
  correlated/systemic shocks, downward or two-sided objectives, limit orders,
  participation caps, adaptive liquidity, anonymity/jamming, or cross-market
  hedging are modeled. The result is not a general theorem about federated
  decision systems.

## Reproduction

```bash
cd /Users/kas/simulations-limit
mechanism-design/.venv-batch/bin/python -m pytest \
  mechanism-design/galanis-market/tests mechanism-design/batch-amm/tests -q
mechanism-design/.venv-batch/bin/python mechanism-design/batch-amm/run_multi.py
```

Raw results: `results/multi_pilot.json` (M=2,000) and `results/multi.json`
(M=20,000). Simulation: `src/batch_amm/multi_market.py`; tests:
`tests/test_multi_market.py`.
