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
capped at the sequential *mean*, far above the sequential *worst seat*. Under the owner's
pseudo-anonymity constraint (traders see only anonymous aggregates, §9) these conclusions
survive almost unchanged — anonymity costs ≤ ~0.1% of trader welfare in aggregation (exactly zero
for symmetric payoffs), needs no extra rounds, and preserves seat-invariance and the damping
requirement — but it adds one new attack: against strict-consistency belief updating, a single
inconsistent order can jam the entire anonymous aggregate (denial-of-aggregation). Both headline
numbers also survive giving traders **limit prices** (§12): the execution advantage ticks up to
2.3–4.3×, while the cheap-manipulation discount is untouched — limits protect the manipulator's
fellow-direction traders, not the market — and a size-best-responding manipulator shows the §6
convention understated batch manipulability.

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
  TWAP LL 0.200 vs 0.139 SEQ). This row assumes per-trader order disclosure; see §9 for the
  pseudo-anonymous version (survives to within ~0.1% of welfare; exactly for Galanis).
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

## 9. Disclosure regimes / pseudo-anonymity

Design constraint (owner): traders must never learn *which trades came from whom* — other traders
see only anonymous aggregates (an attributed ledger may exist for a separate audit layer). The
R ≥ 2 recovery result in §5 assumed FULL disclosure of per-trader orders between rounds, so it was
re-run under three regimes (`Config.disclosure`; sweeps in `results/disclosure.json` and
`results/disclosure_manip.json`, CRN across regimes; plumbing unit-tested in
`tests/test_disclosure.py`):

* **full** — per-trader orders published between rounds (§5's assumption);
* **aggregate** — clearing price + net flow only;
* **price** — clearing price only.

**Regimes (b) and (c) are identical, provably.** Against a deterministic AMM the clearing price
pins down the executed net flow exactly (the LMSR price move is an invertible function of it —
unit-tested), and the net flow pins down the aggregate statistic T = Σ logit(implied target)
given the common-knowledge sizing rule. Publishing the net flow on top of the price adds zero
information; the sweeps produce bit-identical runs. The only disclosure knob with bite is
**attribution**. Note also that a *sequential* market cannot satisfy the constraint at all — each
trade is attributed by timing — so pseudo-anonymity effectively mandates batch clearing.

Belief model under anonymity (behavioral, documented in `envs.py`): Gaussian traders apply a
**mean-field inversion** — read the round-1 aggregate as N identical average traders, invert one
pseudo-signal s̄, hold N copies, swap their own copy for their exact signal (exact when all
signals coincide; unit-tested). Galanis traders do the **exact Bayesian update on the aggregate**:
keep states whose honest quote profile predicts the observed T (tractable because discrete). As
with FULL, only round 1's aggregate is treated as informative.

**Recovery verdict** (honest, damped sizing, R ∈ {2, 3, 5}, M = 20k):

| env | regime | logit err vs full-info | LL vs SEQ (paired) | trader-welfare cost |
|---|---|---|---|---|
| Gaussian N=3  | full | 0.000 | −0.000 | 0 |
| Gaussian N=3  | aggregate/price | 0.114 | +0.0005 ±0.0005 | −0.00005 (0.15%) |
| Gaussian N=10 | aggregate/price | 0.178 | +0.0006 ±0.0005 | −0.00006 (0.13%) |
| Gaussian N=25 | aggregate/price | 0.158 | +0.0004 ±0.0005 | −0.00004 (0.07%) |
| Galanis       | aggregate/price | **0.000** | **+0.0000 (exact)** | **0 (exact)** |

Two findings. (i) **Gaussian: the recovery survives almost entirely, but the residual gap is
permanent, not payable in rounds** — R = 2, 3 and 5 give *identical* results to six decimals (the
model extracts information only from the round-1 aggregate; later flows are re-equilibration).
The plateau is a small logit error (0.11–0.18, the Jensen gap of the mean-field inversion) whose
welfare/log-loss cost is at the edge of detectability (~0.1% of trader profits). So the
rounds-cost of anonymity is: zero extra rounds for ~99.9% of the value, and *no* number of rounds
recovers the last ~0.1% (under myopic play; a more sophisticated updater could mine later-round
flows). (ii) **Galanis: anonymity costs exactly nothing.** The round-1 aggregate reveals the
*bit-count*, which is sufficient for the symmetric payoff X = 1{≥2 bits}; R = 2 anonymous equals
R = 2 full exactly (LL 0.1054, acc 1.000 — unit-tested). Attribution only matters when the payoff
is asymmetric in who holds which signal — a useful test for whether a given market needs
attributed feeds at all.

**Manipulation under anonymity** (bounty sweep re-run under (b), R = 3, batch-competitive):

| env | bounty | Δp_final: full → anon | manip market PnL: full → anon | decision acc: full → anon |
|---|---|---|---|---|
| Gaussian N=3 | 0.50 | +0.275 → +0.242 | −0.0219 → −0.0275 | 0.648 → 0.697 |
| Gaussian N=10 | 0.50 | +0.089 → +0.087 | −0.0012 → −0.0013 | 0.875 → 0.877 |
| Galanis | 0.02 | +0.134 → +0.113 | +0.0005 → −0.0008 | 0.750 → **0.625** |
| Galanis | 0.20 | +0.256 → +0.166 | −0.0036 → −0.0033 | 0.750 → **0.500** |

Opposite signs in the two environments, and the mechanism is instructive:

* **Gaussian: anonymity mildly *dampens* manipulation** (distortion −12%, manipulator cost +25%
  at B = 0.5, N = 3; negligible by N = 10). The soft mean-field inversion spreads the poisoned
  aggregate across N pseudo-copies and every honest trader swaps one copy out for their exact
  signal — the poison is diluted rather than credited in full to a fictitious informant.
* **Galanis: anonymity is *worse* — the manipulator can jam the channel.** Under attribution the
  manipulator's inconsistent order is discarded *individually* and everyone else's information
  still flows: the R = 3 correction restores the 0.750 floor at every bounty. Under anonymity the
  *whole aggregate* becomes unexplainable, the strict-consistency update rejects it, and **no
  information flows at all**: R = 3 anonymous damage collapses to the R = 1 numbers (acc 0.625 at
  B = 0.02, 0.500 at B = 0.20 — below the attributed floor, at *lower* manipulator cost). One
  distorted order converts a price manipulation into a **denial-of-aggregation attack**.
  Anonymous aggregates are a single point of failure for strict-consistency inference; soft
  (regression-style) updates, per-participant net caps, or an audit layer that can selectively
  de-anonymize are the natural mitigations.

Robustness of earlier findings under (b)/(c): batch seat-invariance holds exactly (0 violations at
1e−9 across the full manipulation grid, manipulator rotated through seats with fixed draws; also
unit-tested), and the full-sizing oscillation/divergence is unchanged (per-round |logit p|
profiles match the FULL regime) — **damping is still required**; anonymity neither causes nor
cures it.

Honest caveat: our myopic traders never discounted anyone even under FULL disclosure, so
anonymity's classical defensive cost — honest traders can no longer *condition on the
manipulator's orders* — does not bind in this behavioral model; what is measured is the
inversion-mechanics channel only. At the equilibrium level that cost is real and is bounded by
MANIPULATION.md's phase-2c covert-manipulator results (type uncertainty converts "blind in the
bribed states" into "blurry in all pivotal states"); an anonymous pool is the everyone-unknown
limit of that setting.

## 10. Limitations

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
* **Anonymity model extracts information from the round-1 aggregate only** (mean-field inversion
  for Gaussian, exact aggregate-consistency for Galanis). A rational updater could mine
  later-round net flows for residual information (Ostrovsky-style iterated revelation), so the
  §9 "permanent plateau" is an upper bound on anonymity's aggregation cost for the Gaussian env,
  and the jamming result assumes strict-consistency updating.

## 11. Implications for the proposal-poker engine and bayes-market

Both engines currently run sequential-ish protocols (turn-taking quotes against a curve /
arrival-ordered fills). Under the owner's pseudo-anonymity constraint (traders see only anonymous
aggregates; §9), **sequential protocols are ruled out outright** — a public tape attributes every
trade by timing — so the practical question is not "batch vs sequential" but "how to batch".
**Recommended default: BATCH-LMSR, damped (competitive) sizing, R ≥ 2 rounds, aggregate
disclosure (regime b — equivalently just publish the clearing price), soft aggregate-inversion
beliefs, plus per-participant net caps.** What this buys and costs:

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
4. **Budget at least two batch rounds (or richer bids) — and anonymity barely changes the bill.**
   One-shot batch aggregates markedly worse (logit err grows ~log N); with any disclosure step
   between rounds, round 2 recovers sequential accuracy exactly under attribution and to within
   ~0.1% of trader welfare under anonymity (exactly, for symmetric payoffs like majority — §9).
   No extra rounds are needed for anonymity, and none help past round 2. If rounds are expensive
   (LLM calls), consider bids that carry more than a point order — e.g. demand schedules — the
   natural next mechanism to test.
5. **Watch the cheap-manipulation flank — and, under anonymity, the jamming flank.**
   Uniform-price fills subsidize small pushes (2.4–4.2× cheaper per unit distortion; sometimes
   profitable outright under non-discounting traders). Anonymity adds a second cheap attack:
   where belief-updating is strict/consistency-based, one inconsistent order poisons the whole
   unattributable aggregate and blocks aggregation entirely (Galanis: R=3 floor 0.750 → 0.500,
   §9). Mitigations to test: per-participant net caps, adaptive b, *soft* aggregate inversion
   (the Gaussian-style updater was mildly manipulation-dampening), and an audit layer with
   selective de-anonymization as deterrent.
6. **Auditability scales.** Batch games are exponentially smaller to solve (6 vs 182 infosets at
   N=3, R=1) — equilibrium-checking a production mechanism (the kind of audit MANIPULATION.md
   does) stays tractable at larger N only in the batch design. Anonymity helps here too: the
   anonymous-aggregate game has coarser public histories (net flow instead of order profiles),
   keeping R>1 solves small.

## 12. Limit orders

Every batch result above assumed **pure market orders**: a trader submits a quantity and fills in
full at the uniform clearing price wherever it lands. Two headline findings plausibly depended on
that: the cheap-manipulation discount (§6.2 — honest counterparties absorb the pushed clearing
price via uniform-price fills) and the per-trader execution advantage (§4). This section re-runs
both with **limit prices** (`Config.mech="batch_lmsr_limit"`; sweeps in `scripts/run_limits.py`,
results in `results/limits_*.json`, same SEED/per-N env seeds as every other sweep — CRN against
the §4–§9 baselines; price-only disclosure throughout, per the §9 constraint).

**Mechanism** (`clearing.py`, unit-tested in `tests/test_limit_clearing.py`). An order becomes
(direction, quantity, limit): a buy fills only if the uniform clearing price π ≤ its limit, a
sell only if π ≥ its limit. Clearing is the standard uniform-price call auction against the
curve: π is the unique crossing of φ(D(π)), where φ(X) is the LMSR average execution price of
net flow X and D(π) the eligible net demand (a nonincreasing step function). If the crossing sits
inside a constancy interval of D, all eligible orders fill in full; if it jumps across a limit,
the marginal order's limit **is** the clearing price and marginal orders fill pro-rata. Honest
traders submit their usual competitive-sized quantity with limit = posterior ± slack (slack 0 =
"never trade past my belief"; a loose-limit variant sweeps slack ∈ {0.02, 0.05}); slack = ∞
reproduces the market-order engine **bit-exactly** (asserted on seeded runs, with and without a
manipulator), and a single trader with limit = posterior never executes past their posterior
(asserted). The manipulator keeps the §6 bounty objective, uses limit ±∞ (they want fills), and
now **best-responds in size**: order scaled by γ over a grid {0.25…5}, γ* = argmax of mean
utility (market PnL + bounty·(p_final − ½)); the full grid is recorded. Observers can no longer
read the submitted aggregate off the price — they see only the **executed** net flow — so the
Gaussian mean-field inversion runs on X_exec, and the Galanis exact updater keeps the ω's whose
predicted honest limit-order clearing reproduces X_exec (`reveal_batch_anon_limit`), jam-counting
as in §9. (Implementation note: this work exposed a latent numerical bug in the base batch
engine — when orders net to a pure rounding error, the pro-rata rescale α = X_exec/X divided two
rounding errors and mis-scaled every fill. It never triggers in honest runs — round-1 Galanis
nets are ∝ (2k−3)·ln3 ≠ 0 and Gaussian flow is continuous — but manipulated Galanis R=3 rounds
produce exactly-offsetting flow; the guard (net below 1e−14 ⇒ exact cancellation, full fills at
mid) changes the §6/§9 Galanis R=3 manipulator-PnL entries by up to ~9% (e.g. B=0.2:
−0.0036 → −0.0033) and nothing else; distortion and accuracy columns are unchanged.)

**Q1 — manipulation cost per unit distortion: the 2.4–4.2× discount does NOT close.**
Gaussian, N=3, R=1, B=0.5 (cost = manipulator's market-PnL drop vs honest play, same draws;
`results/limits_manip.json`):

| arm | Δp_final | cost | cost per unit Δp | decision acc (honest 0.831) |
|---|---|---|---|---|
| SEQ seat 0 | +0.216 | 0.0616 | 0.284 | 0.722 |
| SEQ seat 2 (last) | +0.402 | 0.0664 | 0.165 | 0.503 |
| BATCH market orders (γ=1, §6) | +0.158 | 0.0107 | 0.068 | 0.720 |
| BATCH limit, slack 0, γ=1 | +0.154 | 0.0103 | 0.067 | — |
| BATCH limit, slack 0, **γ\*=3** | +0.347 | 0.0473 | 0.136 | 0.556 |
| BATCH limit, slack ∞, **γ\*=3** | +0.352 | 0.0491 | 0.139 | 0.556 |

At the §6 convention (γ=1) the limit-order cost per unit distortion is 0.067 vs sequential
0.165–0.284 — the **2.5–4.2× discount survives to the third digit at every slack**, because the
traders whose limits bind are the *same-direction* ones (buyers priced out by the push), while
the manipulator's actual counterparties — the AMM curve and the contra-directional flow, whose
sell limits sit *below* the pushed price — remain fully eligible. Tight honest limits protect the
like-minded from overpaying; they do not make the manipulator pay more (at slack 0 they pay
marginally *less*: the priced-out honest buys no longer push π up under them). The size
best-response makes it strictly worse: γ*=3 more than doubles the achieved distortion
(+0.35 vs +0.16) at a cost per unit (0.136–0.139) still below the sequential *last seat*, and
utility 0.137 vs 0.077 at γ=1 — the §6 numbers *understated* batch manipulability by fixing
γ = 1. At N=10, B=0.15 the pattern sharpens: γ*=5, Δp_final +0.107 (4× the market-order push)
at **negative** cost — the under-reacting R=1 batch price means a large push toward the
manipulator's own posterior is outright profitable before the bounty. At R=3 the correction
rounds bite as in §6 (market-order c/Δp rises to 0.163 ≈ SEQ-last 0.166) and limits again change
nothing at γ=1 (0.165); γ*=1.5 still buys +0.31 distortion at 0.221 per unit. Galanis (exact):
at γ*, R=1 decision accuracy is 0.500 at *every* bounty ≥ 0.02 under every slack — one notch
below the market-order batch's 0.625 exclusion floor at small B, again because §6 had implicitly
capped the manipulator's size.

**Q2 — the 2.2–4.1× execution advantage survives (marginally improves).**
Gaussian, R=1, honest, per-market execution shortfall vs arrival price (paired, ±≤0.001;
`results/limits_core.json`):

| N | SEQ | market orders | limit slack 0 | fill rate (slack 0) | ΔPnL_total vs market (paired) |
|---|---|---|---|---|---|
| 3 | 0.0492 | 0.0224 (2.2×) | 0.0214 (**2.3×**) | 0.978 | −0.00040 ±0.00001 |
| 5 | 0.0560 | 0.0202 (2.8×) | 0.0191 (**2.9×**) | 0.976 | −0.00052 ±0.00001 |
| 10 | 0.0649 | 0.0190 (3.4×) | 0.0179 (**3.6×**) | 0.976 | −0.00058 ±0.00001 |
| 25 | 0.0729 | 0.0179 (4.1×) | 0.0169 (**4.3×**) | 0.976 | −0.00061 ±0.00000 |

Tight limits drop only 2.2–2.4% of submitted volume (the crowd-priced-out margin), and the
dropped fills are exactly the ones that would have executed above the trader's own belief — so
measured shortfall *falls* and the headline ratio ticks up to 2.3–4.3×. The cost shows up in the
welfare identity, not in execution: the unexecuted information leaves the R=1 price slightly
worse (LL 0.4194 vs 0.4154 at N=3), so paired total PnL is lower by 0.0004–0.0006 per market
(~1–2% of trader profits). Loose limits interpolate smoothly (slack 0.05: fill 99.0%, ΔPnL
−0.0002); the frontier between market orders and tight limits is shallow and monotone.

**Q3 — aggregation: zero extra rounds needed; fixed-R accuracy does not degrade.**
Gaussian log-loss of the final price / |logit error| vs the full-info posterior, by R × slack:

| env, N | R | market orders | slack 0 | slack 0.05 |
|---|---|---|---|---|
| Gaussian 3 | 1 | 0.4154 / 1.52 | 0.4194 / 1.55 | 0.4168 / 1.53 |
| Gaussian 3 | 2 | 0.3602 / 0.114 | 0.3602 / 0.096 | 0.3601 / 0.099 |
| Gaussian 3 | 3, 5 | 0.3602 / 0.114 | 0.3601 / 0.106 | 0.3601 / 0.101 |
| Gaussian 25 | 1 | 0.3714 / 5.16 | 0.3774 / 5.19 | 0.3735 / 5.18 |
| Gaussian 25 | 2–5 | 0.1392 / 0.158 | 0.1391 / 0.107 | 0.1392 / 0.150 |
| Galanis (exact) | 1 | 0.4670, acc 1.000 | identical | identical |
| Galanis (exact) | 2–5 | 0.1054, acc 1.000 | identical | identical |

Unfilled orders do carry information that never reaches the round-1 price — the effect is real
but tiny (N=3: LL +0.004, logit err +0.03 at slack 0). From R = 2 on, the traders who dropped
out re-quote against the new posted price and their information arrives anyway: recovery is
complete at R = 2 for every slack, exactly as with market orders — **R_needed(limit) =
R_needed(market) = 2, no extra rounds**. At R ≥ 2 tight limits even *shave* the §9 anonymity
plateau (N=25 logit err 0.158 → 0.107: the executed flow is closer to the mean-field observer's
model when the overshooting tail is truncated). Galanis is exact and bit-identical in accuracy
at every R: the executed-flow partition still reveals the bit-count.

**Q4 — jamming: the attack surface narrows, the funded attack is unchanged.**
Galanis, R=3, price-only disclosure, manipulated (`results/limits_jam.json`; honest runs jam 0
at every slack and stay at acc 1.000):

| bounty | market orders acc / jams | limit slack 0, γ=1 | limit slack 0, γ* | limit slack ∞, γ* |
|---|---|---|---|---|
| 0.02 | 0.625 / 24 | 0.625 / 24 | 0.625 / 21 (γ\*=1.5) | 0.625 / 21 |
| 0.05 | 0.625 / 24 | 0.625 / 24 | 0.500 / 21 (γ\*=1.5) | 0.500 / 21 |
| 0.20 | 0.500 / 24 | 0.500 / 24 | 0.500 / 24 (γ\*=5) | 0.500 / 24 |

The structural hope — an inconsistent anonymous order that can't fill can't stall clearing — is
half right. Under limit clearing an order only distorts the executed flow if it *fills*: a
"free" jam via an unfillable order is now impossible by construction (a no-fill leaves X_exec at
its honest value, which is always explainable — the honest rows jam zero), whereas under market
orders *every* submitted order enters X. So jamming now requires taking, and paying for, a real
position. But the §9 attacker already did exactly that: their ±∞-limit order fills in full
(fill rate 1.000 across the grid), the executed aggregate is unexplainable, and the
strict-consistency update freezes — same jams, same accuracy collapse, at every slack, γ for γ.
Honest tight limits never pin π at an honest posterior in this environment (the pooling that
would let a manipulated outcome masquerade as honest does not materialize), so **limit
protection does not restore aggregation against a funded manipulator** — and the attack still
*pays* (utility +0.003 to +0.074 across bounties). The §9 mitigations (soft updates, net caps,
audit-layer de-anonymization) remain the operative ones.

**Verdict on the two headline numbers.** (1) The 2.4–4.2× cheap-manipulation batch discount
**survives** limit orders essentially unchanged (2.5–4.2× at the §6 convention, every slack) —
and letting the manipulator best-respond in size shows the original number was, if anything, an
*understatement* of batch manipulability (double the distortion at sequential-last-seat unit
cost; outright profitable at N=10). Limit prices protect the manipulator's fellow-direction
traders, not the market. (2) The 2.2–4.1× per-trader execution advantage **survives and ticks
up** (2.3–4.3×) at a ~1–2%-of-profits aggregate welfare cost paid through slightly worse one-shot
aggregation; recovery still needs exactly two rounds. Neither headline reverses; the §11
recommendation is unchanged, with one addition: offering limit orders is cheap execution
insurance for honest participants, but it is **not** a manipulation or jamming defense.

## Reproduction

```bash
cd mechanism-design
python3.11 -m venv .venv-batch && .venv-batch/bin/pip install numpy scipy pytest
.venv-batch/bin/pip install -e galanis-market --no-deps -e batch-amm
.venv-batch/bin/python -m pytest batch-amm/tests -q          # 62 tests
.venv-batch/bin/python batch-amm/scripts/run_sweeps.py       # ~40 s, writes results/*.json
.venv-batch/bin/python batch-amm/scripts/run_disclosure.py   # ~6 s, disclosure-regime sweeps
.venv-batch/bin/python batch-amm/scripts/run_limits.py       # ~100 s, limit-order sweeps (§12)
.venv-batch/bin/python batch-amm/scripts/cfr_batch.py        # ~4 s, writes results/cfr_batch.json
```

Raw results: `results/core.json` (honest sweeps + paired diffs), `results/manip.json`
(bounty × seat sweeps), `results/seats.json` (rotation-averaged seat PnL),
`results/disclosure.json` / `results/disclosure_manip.json` (pseudo-anonymity sweeps, §9),
`results/limits_core.json` / `results/limits_manip.json` / `results/limits_jam.json`
(limit-order sweeps, §12), `results/cfr_batch.json` (equilibrium check). Sim engine:
`src/batch_amm/engine.py` (mechanism definitions in the module docstring),
`src/batch_amm/clearing.py` (limit-order uniform-price clearing), `src/batch_amm/envs.py`
(environments and the belief/inversion model, incl. the anonymous-aggregate updaters).
