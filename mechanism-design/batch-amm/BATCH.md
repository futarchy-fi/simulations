# Batch-cleared vs sequential trading against an AMM in small decision markets

**Question.** Does batch clearing give "price improvement" over sequential trading against an AMM?
The motivating claim: *sequential traders each walk the curve and pay their own slippage, while a
netted batch clears at one uniform price — offsetting flow trades at mid, only net flow moves the
curve — so batch leaks less value to the curve, kills intra-round ordering games, and scales better
with N.*

**Method.** Three mechanisms run on the same trader population and the *same environment draws*
(common random numbers, paired differences), swept over N ∈ {3, 5, 10, 25} and R ∈ {1, 3} rounds.
Traders are **myopic-Bayes** programmed agents (the `galanis-market/myopic.py` model: quote your
posterior; trades are inverted into signals by a public that assumes everyone is honest-myopic).
This is explicitly a **behavioral first pass, not equilibrium** — the one-shot batch game is
re-solved to equilibrium with CFR+ as a check (below). Gaussian results use M = 20,000 Monte Carlo
reps (95% CIs reported as ±); Galanis results are exact 8-state enumerations (no MC error).

**Verdict in one paragraph.** The claim is *right about execution and ordering, wrong about
aggregate leakage, and mixed on manipulation*. In an LMSR, aggregate trader welfare is a pure
function of the final price (identity below), so batching cannot reduce the total leaked to the
curve except by changing the final price — and with myopic traders it changes it for the *worse*
at R = 1 (batch forfeits intra-round inference) and not at all at R ≥ 2 (competitive sizing lands
on exactly the sequential final price). What batching *does* deliver: per-trader execution
shortfall vs arrival price falls 2.2× (N=3) → 4.1× (N=25) at R = 1, honest ordering rents
(first seat earns 3.6× the last seat at N=3, **69×** at N=25) are zeroed exactly, and the
manipulation *seat lottery* — including the catastrophic bribed-last-mover of
`mechanism-design/MANIPULATION.md` phase 2a — is removed by construction. The price paid: with
non-discounting traders, a given bounty buys price distortion **2.4–4.2× more cheaply** in batch
(uniform-price fills give the manipulator average-price execution), though worst-case damage is
capped at the sequential *mean*, far above the sequential *worst seat*.

---

## 1. Mechanisms (precise definitions)

All markets trade one binary security paying X ∈ {0,1}, start at p = 0.5, and use an LMSR with
liquidity b matched across arms (Gaussian env: b = 0.1; Galanis env: b = 0.01, matching the CFR+
setup in MANIPULATION.md). LMSR conventions as in `galanis-market/lmsr.py`: moving the marginal
price p→p′ costs C(p′) − C(p) with C(p) = −b·ln(1−p) and yields b·(logit p′ − logit p) shares.

**SEQ-LMSR** (the galanis-market protocol). Each round, traders act one at a time in fixed seat
order. On their turn a trader moves the price to their target, paying the curve integral.

**BATCH-LMSR.** Each round, every trader simultaneously sees the posted price p and submits a
market order x_i = s·b·(logit t_i − logit p), where t_i is their target and s a sizing factor.
Clearing rule (unit-tested in `tests/test_clearing.py`):

* orders are **netted**: X = Σ x_i; the AMM absorbs only the net flow, logit p′ = logit p + X/b;
* one **uniform clearing price** π = [C(p′) − C(p)] / X — the average price of the net execution
  along the curve; π = p (mid) when X = 0, and π is continuous there;
* every order fills in full at π. Cash conserves exactly: Σ π·x_i = C(p′) − C(p), so offsetting
  flow crosses trader-to-trader at π and never touches the curve. If p′ would leave the price
  bounds, all orders are scaled pro-rata (rare; tested).

Sizing s: **full** (s = 1, the literal sequential trade; single-trader batch reduces *exactly* to
sequential — tested) and **competitive** (s = 1/N: each trader expects to be one of N pushing the
same way; equals full at N = 1, and N identical targets clear exactly at the target). Full sizing
is unstable: N like-minded myopic traders overshoot jointly (net logit move = sum of individual
moves) and re-correction oscillates divergently across rounds — visible below; **competitive is
the headline batch arm**.

**BATCH-KYLE** (bridge case). Same batch protocol against a linear conditional-expectation maker:
depth matched to the LMSR's local depth λ = p(1−p)/b; orders x_i = s·(t_i − p)/2λ (myopic optimum
against own linear impact); a single uniform price π = p′ = p + λX for all fills *and* the maker's
position. Unlike the curve, the linear maker has no convexity revenue.

**Manipulator** (one seat, both envs): utility = market PnL + bounty·(p_final − 0.5), treated
myopically (each move priced as if final). Against the LMSR the optimal quote solves
b(q−t)/(t(1−t)) + B = 0, closed form in `engine.py`, grid-verified. Honest traders do **not**
discount the manipulator — the deliberate behavioral contrast with the CFR+ equilibrium results.

## 2. Environments

* **Gaussian**: v ~ N(0,1), s_i = v + ε_i, ε_i ~ N(0,1), security pays 1{v > 0}. Full-information
  posterior is closed-form; myopic-Bayes sequential play is fully revealing (final price = full-info
  posterior, up to price-bound saturation in extreme tails — tested).
* **Galanis "Easy"** (`t3s111y2`): ω uniform on {0,1}³, trader i sees bit i, X = 1{≥2 bits}.
  Quotes capped to [0.1, 0.9] mirroring the tabular grid of the CFR+ results, b = 0.01 — so scales
  are directly comparable with MANIPULATION.md (honest SEQ here: log err 0.1054, decision acc
  1.000, total trader profit +0.0059 — the CFR+ equilibrium totals are 0.105 / 1.000 / +0.0059).
  SEQ R=1 reproduces `myopic.py`'s trajectory exactly (tested).

## 3. The welfare identity that reframes the claim

For any cost-function market maker, the AMM's cash revenue over a session telescopes:
Σ trades (C(p_k) − C(p_{k−1})) = C(p_final) − C(p_0), **independent of the path**. For the LMSR
specifically, total trader PnL against the maker reduces to

  Σ_i PnL_i = −MM PnL = b·[ln 2 − LogLoss(p_final)],

where LogLoss is the log score of the final price against the realized outcome. (Verified to
machine precision in every run; e.g. Gaussian N=3 SEQ: 0.1·(ln2 − 0.3597) = +0.03335 = measured.)

Consequences: **aggregate** "value leaked to the curve" is *not a mechanism knob* — it depends only
on where the price ends. Batch clearing can only change (i) the final price (via aggregation
dynamics), (ii) the *distribution* of welfare across traders (ordering rents, execution), and
(iii) the cost/geometry of manipulation. That is exactly what the measurements show.

## 4. Price improvement (execution) and its N-scaling

Execution shortfall = cash paid above the trader's arrival price (turn-open price in SEQ, posted
price in BATCH), summed over traders, per market. Honest population, Gaussian, paired vs SEQ:

| N | SEQ leak | BATCH-comp leak (R=1) | improvement | BATCH-comp leak (R=3) | Δ vs SEQ (R=3) |
|---|---|---|---|---|---|
| 3  | 0.0492 ±0.0006 | 0.0224 | **−55% (2.2×)** | 0.0324 | −34% |
| 5  | 0.0560 ±0.0005 | 0.0202 | **−64% (2.8×)** | 0.0440 | −21% |
| 10 | 0.0650 ±0.0005 | 0.0190 | **−71% (3.4×)** | 0.0699 | **+8% (worse)** |
| 25 | 0.0729 ±0.0005 | 0.0179 | **−76% (4.1×)** | 0.1106 | **+52% (worse)** |

(All paired CIs ≤ ±0.001; `results/core.json`.) So the headline number: **batch cuts per-market
execution shortfall by 2.2×–4.1×, and the improvement grows with N** — at R = 1. Two honest
caveats. First, batch-competitive traders also *trade ~10× less volume*; per unit volume batch
execution is not better (netting pays only when flows disagree; with signals correlated through v,
flow is mostly one-directional). Second, at R = 3 with larger N the ranking *reverses*: sequential
trading spreads the same repricing over many small, individually-cheap steps, while batch does it
in one large netted move — at N = 25 sequential execution is 52% cheaper. **Netting wins when
disagreement is high relative to shared direction; fine-grained sequential walking wins when the
market is mostly agreeing.**

Aggregate welfare (= −MM PnL, by the identity): batch-competitive at R = 3 lands on *exactly* the
sequential final price, so ΔPnL_total = 0.00000 ±0.00000 in every N — the mechanisms are
aggregate-welfare-identical. At R = 1 batch aggregate welfare is *lower* (e.g. N=25: +0.0322 vs
+0.0554; the AMM keeps the difference) because batch aggregates worse in one shot — next section.

## 5. Aggregation

Log-loss of final price vs realized outcome, and |logit error| vs the full-information posterior
(Gaussian; SEQ is fully revealing so its logit error ≈ 0):

| N | SEQ LL | BATCH-comp LL (R=1) | BATCH-comp logit err (R=1) | BATCH-comp LL (R=3) | BATCH-full LL (R=1) |
|---|---|---|---|---|---|
| 3  | 0.360 | 0.415 | 1.52 | 0.360 (err 0.000) | 0.374 (err 0.68) |
| 5  | 0.289 | 0.394 | 2.44 | 0.289 (err 0.000) | 0.321 (err 1.29) |
| 10 | 0.218 | 0.380 | 3.71 | 0.218 (err 0.000) | 0.290 (err 1.89) |
| 25 | 0.139 | 0.371 | 5.16 | 0.139 (err 0.000) | 0.244 (err 1.94) |

* **R = 1: batch is strictly worse at aggregation, and the gap grows with N.** Sequential trading
  gets intra-round inference for free — each trader conditions on everything predecessors revealed;
  the last quote is the full-information posterior. A single batch collects everyone's *prior-based*
  quote: competitive sizing averages the individual log-odds (under-reaction, logit err ~ log N);
  full sizing sums them (over-reaction, since signals are correlated through v).
* **R ≥ 2 heals it completely** (competitive sizing): round 1's disclosed orders reveal all
  signals, round 2 clears exactly at the shared posterior. Final-price aggregation is then
  *identical* to sequential; only TWAP retains a penalty from the bad first round (N=25 R=3:
  TWAP LL 0.200 vs 0.139 SEQ).
* **Full sizing must not be iterated**: overshoot re-corrections oscillate divergently
  (N=5 R=3: LL 0.94 vs 0.29; Galanis R=3: decision accuracy collapses to 0.25). BATCH-KYLE full
  R=3 is catastrophic (traders lose −153 per market at N=25 to their own oscillation). Any real
  batch-against-AMM protocol needs damped sizing or equivalent (participation limits, per-round
  net caps).
* Galanis (exact): same story — SEQ R=1 hits the myopic ideal (LL 0.105, acc 1.000); one-shot
  batch is worse (comp 0.467, full 0.225) but *decision* accuracy stays 1.000 (prices land on the
  right side of 0.5); R=3 competitive matches SEQ exactly (0.105, acc 1.000).

## 6. Manipulation: cost-to-distort and seat effects

One trader holds a bounty B·(p_final − 0.5). SEQ is swept over every seat (manipulator = same
trader/signal rotated through the order with the same draws); BATCH is run at first and last seat
(outcomes match to 1e−10; exact seat-invariance also unit-tested). "Cost" = manipulator's market
PnL drop vs their honest play on the same draws.

**Gaussian, N=3, R=1, B=0.5** (per-seat SEQ vs BATCH; `results/manip.json`):

| arm | Δp_final | manip market PnL (vs honest) | cost per unit Δp | decision acc (honest 0.831) |
|---|---|---|---|---|
| SEQ seat 0 | +0.216 | −0.042 (−0.062) | 0.285 | 0.722 |
| SEQ seat 1 | +0.309 | −0.056 (−0.065) | 0.210 | 0.602 |
| SEQ seat 2 (last) | +0.403 | −0.061 (−0.066) | 0.165 | 0.503 |
| BATCH-LMSR (any seat) | +0.158 | −0.001 (−0.011) | **0.068** | 0.720 |
| BATCH-KYLE (any seat) | +0.124 | −0.006 (−0.010) | 0.083 | 0.675 |

At N=10, B=0.15 the SEQ seat gradient is a full **5×** (Δp_final +0.040 first seat → +0.199 last
seat; accuracy 0.897 → 0.846), while batch sits at +0.026, seat-independent.

**Galanis (exact), R=1** — decision accuracy (honest = 1.000):

| bounty | SEQ seat 0 | SEQ seat 1 | SEQ seat 2 (last) | BATCH-LMSR | BATCH-KYLE |
|---|---|---|---|---|---|
| 0.02 | 0.750 | 0.750 | **0.500** | 0.625 | 0.750 |
| 0.05 | 0.750 | 0.750 | **0.500** | 0.625 | 0.750 |
| 0.20 | 0.750 | 0.750 | **0.500** | 0.500 | 0.750 |

At R=3 batch holds the 0.750 information-exclusion floor at every bounty (honest re-quotes correct
the push each round) while the SEQ last seat stays broken at 0.500.

Findings:

1. **The sequential seat lottery is real and batch removes it.** SEQ last-mover distortion is
   1.9× (Gaussian N=3) to 5× (N=10) the first seat's, and in the Galanis env the bribed last
   mover breaks the decision to coin-flip at *any* bounty ≥ 0.02 — the same last-mover
   catastrophe as MANIPULATION.md phase 2a, reproduced here under behavioral (non-discounting)
   play. Batch outcomes are provably and measurably seat-invariant; worst-case manipulation
   damage drops from the sequential *worst seat* to roughly the sequential *mean*.
2. **But per unit of price distortion, batch manipulation is 2.4–4.2× cheaper.** Uniform-price
   clearing fills the manipulator at the *average* execution price of the net flow — honest
   contra-flow subsidizes the push. At small bounties the batch manipulator's market book is even
   *profitable* (Galanis B=0.02: +0.0002; the sequential manipulator always pays). Hanson's
   "manipulator as subsidy donor" is weakest precisely in the batch design.
3. **The linear maker is the worst of both**: BATCH-KYLE at R=3, B=0.5 reaches Δp_final +0.42 and
   acc 0.508 while costing the manipulator less than any LMSR arm, and honest traders *lose*
   money — repeated push-correct cycles bleed a static-λ linear maker and everyone trading with it.
   The LMSR's convexity is doing real defensive work.

## 7. Ordering games (honest traders)

Rotation-averaged per-seat expected PnL, honest SEQ, Gaussian (same signal multiset rotated
through all seats; `results/seats.json`):

| N | seat 0 | seat 1 | last seat | spread | spread / avg trader profit |
|---|---|---|---|---|---|
| 3  | +0.0193 ±0.0004 | +0.0088 | +0.0053 | 0.0140 | 1.3× |
| 5  | +0.0196 | +0.0090 | +0.0028 | 0.0168 | 2.1× |
| 10 | +0.0194 | +0.0089 | +0.0010 | 0.0184 | 3.9× |
| 25 | +0.0194 | +0.0089 | +0.0003 | 0.0191 | 8.6× |

Being first is worth ~0.019 per market *regardless of N* (you trade against the prior at maximal
mispricing); later seats inherit an already-informative price. As N grows the *total* pie stays
similar, so ordering becomes the dominant determinant of a trader's income: at N=25 the first seat
earns **69× the last seat**, 8.6× the average. In the Galanis structure the sign reverses (last
seat +0.0029 vs first +0.0013 — the capped 0.9 grid leaves the profitable certainty-move to the
finisher): ordering rents are structure-dependent in sign, but always present in SEQ. **Batch
zeroes the seat spread exactly (≤ 1e−18, machine zero) by construction** — same aggregate welfare
(R ≥ 2), equal split.

## 8. Equilibrium check (stretch): CFR+ on the one-shot batch game

`scripts/cfr_batch.py` solves the simultaneous-move BATCH-LMSR game on the Galanis structure
exactly (9-point target grid, full sizing — order size is endogenous at equilibrium since a small
order is just a near-mid target; b = 0.01, same discretisation as the sequential CFR+ in
MANIPULATION.md). CFR+ with RM+, linear averaging, 4000 iters; NashConv ≤ 4.2e−8 (coarse
correlated equilibrium caveat as usual). Results (`results/cfr_batch.json`):

| bounty | LL final | decision acc | Δmean price | NashConv |
|---|---|---|---|---|
| 0.00 | 0.171 | 1.000 | — | 1.1e−8 |
| 0.05 | 0.422 | 0.750 | +0.120 | 2.8e−8 |
| 0.20 | 0.422 | 0.750 | +0.120 | 4.2e−8 |

* **The headline results are not myopic artifacts.** At equilibrium the one-shot batch still
  aggregates worse than the sequential game's equilibrium (LL 0.171 vs 0.105) — rational traders
  shade their sizes (equilibrium play beats myopic full sizing's 0.225 but cannot recreate
  intra-round inference. The gap is structural, not behavioral.) Under bounties, equilibrium
  batch degrades to the 0.750 information-exclusion floor and *stops there* — no seat exists to
  produce the 0.500 last-mover collapse; and unlike the myopic sweep (which hit 0.500 at B = 0.2
  because honest traders never discount), equilibrium honest traders discount the manipulator and
  hold 0.750. Batch + rational opponents = the phase-1 exclusion floor, seat-free.
* **Solve cost collapses**: the simultaneous game has **6 infosets** (2 per player) vs **182** for
  the sequential game at the same grid and R=1 — and sequential infosets grow as
  cells·(grid)^(seats before you), i.e. exponentially in N, while batch grows linearly (one
  infoset per signal per round; with per-round net-flow disclosure the R>1 game stays small).
  Each CFR+ solve here takes ~1.2 s.

## 9. Limitations

* **Myopic-Bayes traders, no discounting of manipulators.** The main sweeps are behavioral: honest
  traders never suspect anyone. Equilibrium honest play (MANIPULATION.md, and §8) discounts
  manipulators and changes manipulation numbers, mostly in batch's favor. Conversely myopic
  sequential play is *fully revealing*, which flatters SEQ's aggregation; LLM/human traders fall
  short of that (Galanis 2026).
* **Matched-b caveat.** "b matched across arms" equalizes the *maker's* pricing rule, not risk:
  the batch maker faces one netted fill per round (less adverse-selection exposure per unit
  depth), so a real operator could quote batch at higher b — which would improve every batch
  number here except manipulation cost. The Kyle λ is matched only locally at the posted price
  and is static within a round; an adaptive-λ maker would resist §6.3.
* **Sizing conventions are behavioral choices.** Full vs competitive bracket the reasonable
  myopic behaviors; equilibrium sizing (§8) lands between them. The instability of iterated
  full-sizing batch is a property of naive re-submission, not of batch clearing per se.
* **Price bounds**: Gaussian quotes clip at 1e−4 (tail saturation loses information in extreme
  reps); Galanis quotes cap at 0.1/0.9 for CFR-grid comparability. Threshold *locations* inherit
  these choices; qualitative rankings were spot-checked as robust to the cap.
* Bounty is common knowledge to no one (honest traders can't react) — the opposite pole from
  MANIPULATION.md's common-knowledge equilibria; reality is between (its phase 2c).
* TWAP here averages round-end prices (R points), so R=1 TWAP = final price.

## 10. Implications for the proposal-poker engine and bayes-market

Both engines currently run sequential-ish protocols (turn-taking quotes against a curve /
arrival-ordered fills). What switching to per-round batch clearing would buy:

1. **Kills the decision-reading attack surface at its worst point.** MANIPULATION.md phase 2a
   showed *any* bribed last mover — even uninformed — breaks a final-price decision rule
   (acc 0.5 < the 0.75 no-entry floor). Batch has no last mover inside a round; the myopic sweeps
   and the CFR+ check both show batch bottoming at the information-exclusion floor instead of the
   last-mover collapse. Batch clearing and a TWAP-style decision statistic are complementary
   fixes (batch removes intra-round ownership; TWAP removes across-round final-tape ownership).
2. **Removes honest ordering rents — an incentive-design win, not a welfare win.** In a
   sequential engine the first seat earns up to 8.6× the average trader (N=25); that pays agents
   for latency races and seat camping, not for information. Batch redistributes exactly that rent
   at zero aggregate cost (identity, §3). For bayes-market, where participants arrive
   asynchronously, per-epoch batch windows would decouple reward from arrival order.
3. **Do not sell batch as "less value leaked to the AMM".** Aggregate leak is pinned to the final
   price by the cost-function identity; at matched b and matched information, batch saves the
   *traders as a group* nothing. The honest pitch is per-trader execution (2–4× better shortfall
   vs arrival, §4) and fairness, not subsidy savings.
4. **Budget at least two batch rounds (or richer bids).** One-shot batch aggregates markedly worse
   (logit err grows ~log N); with a disclosure step between rounds, round 2 recovers sequential
   accuracy exactly. If rounds are expensive (LLM calls), consider bids that carry more than a
   point order — e.g. demand schedules — which is the natural next mechanism to test.
5. **Watch the cheap-manipulation flank.** Uniform-price fills subsidize small pushes (2.4–4.2×
   cheaper per unit distortion; sometimes profitable outright under non-discounting traders). If
   the engine's traders are naive LLM agents rather than equilibrium discounters, batch trades a
   rare catastrophic failure for a chronic cheap one. Mitigations to test: per-participant net
   caps, adaptive b, disclosure of per-seat net positions.
6. **Auditability scales.** Batch games are exponentially smaller to solve (6 vs 182 infosets at
   N=3, R=1) — equilibrium-checking a production mechanism (the kind of audit MANIPULATION.md
   does) stays tractable at larger N only in the batch design.

## Reproduction

```bash
cd mechanism-design
python3.11 -m venv .venv-batch && .venv-batch/bin/pip install numpy scipy pytest
.venv-batch/bin/pip install -e galanis-market --no-deps -e batch-amm
.venv-batch/bin/python -m pytest batch-amm/tests -q          # 28 tests
.venv-batch/bin/python batch-amm/scripts/run_sweeps.py       # ~40 s, writes results/*.json
.venv-batch/bin/python batch-amm/scripts/cfr_batch.py        # ~4 s, writes results/cfr_batch.json
```

Raw results: `results/core.json` (honest sweeps + paired diffs), `results/manip.json`
(bounty × seat sweeps), `results/seats.json` (rotation-averaged seat PnL),
`results/cfr_batch.json` (equilibrium check). Sim engine: `src/batch_amm/engine.py`
(mechanism definitions in the module docstring), `src/batch_amm/envs.py` (environments and the
belief/inversion model).
