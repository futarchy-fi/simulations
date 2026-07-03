# Corruption resistance of small decision markets: manipulator sweeps at equilibrium

**Question.** How much outside incentive (a "bounty" paid to one trader for a particular market outcome) can a small decision market absorb before its decisions degrade? Hanson's manipulation-as-subsidy argument ([hanson.gmu.edu/biashelp.pdf](https://hanson.gmu.edu/biashelp.pdf)) says manipulators should not scare us: a trader who pushes prices for non-informational reasons is a predictable noise trader, and their expected losses become a subsidy that pays informed traders to correct the price. We test that claim *at equilibrium* — every trader, including the manipulator, plays a best response — in three mechanism designs, sized like the small markets an agent collective might actually run as an oversight layer: a handful of traders, each holding a private piece of the evidence, with an explicit decision rule reading the final price.

**Why this matters for markets-as-oversight.** If prediction/decision markets are to serve as an aggregation and oversight layer for collectives of AI agents, the first-order attack is exactly this one: some principal outside the market gains from decision D and offers one of the traders a side payment conditional on D. The sweeps below measure, mechanism by mechanism, (i) how large that side payment must be before equilibrium decisions actually change, (ii) what the corruption costs the briber and the bribed, and (iii) whether the losses really do end up subsidising the honest traders, as Hanson predicts.

These are **rational-equilibrium results** (tabular CFR+, NashConv ≤ 6×10⁻⁴, usually ≤ 3×10⁻⁵). The companion empirical question — what *behavioral LLM traders* do under the same bounties, per Galanis 2026 ([arXiv:2604.20050](https://arxiv.org/abs/2604.20050)) and Ouyang & Sui 2026 ([arXiv:2604.18373](https://arxiv.org/abs/2604.18373)) — is a separate planned experiment; equilibrium play is the *upper bound* on trader rationality, and the Galanis results suggest LLM traders fall well short of it on hard structures.

## Common setup

All three games share the information structure of Galanis's "Easy" treatment:

* Nature draws ω = (d_a, d_b, d_c) uniformly from {0,1}³ (8 states).
* Three traders; trader *i* privately observes bit *i* plus the full public trade history. Three rounds, fixed rotation, one quote each.
* Trades execute against an automated market maker; the security pays out on the relevant metric at resolution.
* **Player 0 is the manipulator**: their utility = market P&L + an outside bonus that depends on the market outcome. Players 1–2 are ordinary Bayesian traders. Everyone's strategy — including how players 1–2 *discount the manipulator's trades* — is solved jointly by CFR+.
* Bonuses are denominated in units of the security's notional (payout = 1). For scale: an honest trader's baseline expected profit per market is ≈ 0.001–0.007, and the LMSR (b = 0.01) cost of pushing the price 0.5 → 0.9 against the truth is ≈ 0.016 (CFMM K = 0.001: ≈ 0.063).

| Mechanism | Market structure | Decision rule | Manipulator's bonus |
|---|---|---|---|
| **Galanis LMSR** (`galanis-market`) | one LMSR market on X = 1{≥2 of 3 bits} | proxy: act iff final price ≥ 0.5 | `bonus · (p_final − 0.5)` (wants high price) |
| **Hanson conditional** (`hanson-conditional`) | two conditional LMSR markets, M\|A and M\|B; losing market refunds | implement policy with higher final price (tie → A) | flat `bonus` iff A implemented |
| **MetaDAO CFMM** (`metadao-cfmm`) | same as Hanson but constant-product AMM pools (K = 0.001) | same | flat `bonus` iff A implemented |

Metrics: M(A) = 1{≥2 yes-bits}, M(B) = 1{≥1 yes-bit}. **Decision accuracy** = P[the implemented policy's metric realises as 1], averaged over ω; the full-information optimum is 0.875. Sweep scripts: `*/scripts/manipulator_sweep.py`; raw results: `galanis-market/results/manipulator_sweep_t3s111y2.json`, `hanson-conditional/results/hanson_manipulator_sweep.json`, `metadao-cfmm/results/metadao_manipulator_sweep.json`. Solved 2026-07-02/03 with 400 (Galanis) / 200 (Hanson, MetaDAO) CFR+ iterations per bonus level.

## Headline comparison

Deltas are vs. each mechanism's own bonus = 0 baseline. "Manipulator net market cost" is the manipulator's *trading* loss (their outside bonus receipts excluded). "Informed Δ" is players 1+2's combined profit change — the Hanson subsidy, if it exists.

| | Galanis LMSR | Hanson conditional | MetaDAO CFMM |
|---|---|---|---|
| Baseline quality (bonus = 0) | log err 0.105; decision acc 1.000 | decision acc 0.752 | decision acc **0.875** (= optimum) |
| Smallest bonus with visible damage | between 0.02 and 0.05 (sharp threshold) | 0.05 (0.02 *improves* acc to 0.803) | **0.005** (acc −0.10 immediately) |
| Quality at worst swept point | log err 0.399; acc 0.750 (bonus ≥ 0.1) | acc 0.679 (bonus ≥ 0.05, saturates) | acc 0.685 (bonus = 0.2) |
| Decision-flip rate at saturation | 0.25 (2 of 8 states flip) | 0.28 mean \|ΔP(A)\| per state | 0.43 mean \|ΔP(A)\| per state |
| Can the briber actually buy decision A? | n/a (directional price target) | no: P(A) 0.45 → 0.54 at bonus 0.2 | no: P(A) 0.125 → 0.47 at bonus 0.2 |
| Manipulator net market cost (saturation) | −0.0035 (from +0.0012 to −0.0022) | −0.0044 (from +0.0012 to −0.0033) | −0.0179 (from +0.0071 to −0.0108) |
| Informed traders' profit Δ | +0.0005 (peak +0.0034 mid-transition) | +0.0040 | +0.0138 |
| Share of manipulator loss captured by informed | **15%** (54% mid-transition; rest returns to the market maker) | **91%** | **77%** |

### Per-mechanism sweeps

**Galanis LMSR** (player 0 paid to push the price up; 9-point price grid; NashConv ≤ 5×10⁻⁵ every row):

| bonus | mean log err | decision acc | manip market P&L | informed P&L |
|---|---|---|---|---|
| 0.00 | 0.105 | 1.000 | +0.00124 | +0.00463 |
| 0.01 | 0.105 | 1.000 | +0.00124 | +0.00463 |
| 0.02 | 0.106 | 0.999 | +0.00081 | +0.00506 |
| 0.05 | 0.359 | 0.796 | −0.00402 | +0.00723 |
| 0.10 | 0.399 | 0.750 | −0.00511 | +0.00805 |
| 0.20 | 0.399 | 0.750 | −0.00223 | +0.00517 |
| 1.00 | 0.399 | 0.750 | −0.00223 | +0.00517 |

The damage is surgical: only the 4 states where player 0's bit is the *pivotal* signal (ω = b, c, f, g) collapse — their equilibrium price goes to ≈ 0.50, which is exactly the Bayesian posterior *with player 0's bit deleted*. The other 4 states stay pinned at 0.10/0.90. Two of the broken states (f, g: truth = no) flip the ≥ 0.5 decision proxy; decisions degrade from perfect to 0.75.

**Hanson conditional** (player 0 paid iff A implemented; NashConv ≤ 3×10⁻⁵):

| bonus | decision acc | mean \|ΔP(A)\| | manip market P&L | informed P&L | expected bonus received |
|---|---|---|---|---|---|
| 0.000 | 0.752 | — | +0.00117 | +0.00204 | 0 |
| 0.005 | 0.753 | 0.33 | +0.00074 | +0.00262 | 0.0029 |
| 0.020 | **0.803** | 0.30 | +0.00116 | +0.00232 | 0.0088 |
| 0.050 | 0.679 | 0.28 | −0.00325 | +0.00605 | 0.0268 |
| 0.100 | 0.679 | 0.28 | −0.00325 | +0.00606 | 0.0536 |
| 0.200 | 0.679 | 0.28 | −0.00325 | +0.00606 | 0.1071 |

**MetaDAO CFMM** (same bonus structure; NashConv ≤ 5.7×10⁻⁴):

| bonus | decision acc | mean \|ΔP(A)\| | manip market P&L | informed P&L | expected bonus received |
|---|---|---|---|---|---|
| 0.000 | **0.875** | — | +0.00713 | +0.00746 | 0 |
| 0.005 | 0.776 | 0.38 | +0.00392 | +0.01095 | 0.0025 |
| 0.020 | 0.756 | 0.45 | +0.00249 | +0.00924 | 0.0115 |
| 0.050 | 0.799 | 0.37 | +0.00312 | +0.00901 | 0.0248 |
| 0.100 | 0.810 | 0.29 | +0.00381 | +0.00837 | 0.0418 |
| 0.200 | 0.685 | 0.43 | −0.01077 | +0.02123 | 0.0942 |

## Interpretation

### Does manipulation-as-subsidy hold at equilibrium?

**The transfer half of Hanson's claim holds; the accuracy half fails.** In all three mechanisms, once the bounty is large enough to change the manipulator's play, the manipulator loses money in the market and the informed traders' profits rise — in the two decision-market mechanisms the informed side captures 77–91% of the manipulator's trading losses, close to a pure transfer (in single-market LMSR, most of it instead flows back to the market maker as reduced subsidy payout). So yes: at equilibrium, manipulation is a subsidy to informed traders.

But the subsidy does not buy back decision quality. Galanis-LMSR accuracy falls from 1.00 to 0.75, Hanson from 0.75 to 0.68, MetaDAO from 0.875 to 0.685 — *while* the informed traders pocket the subsidy. The reason is visible in the per-state prices: the equilibrium defence against a known manipulator is not "trade against them harder and correct the price" (Hanson's mechanism), it is **discount their trades entirely**. And in a small market the manipulator is not a pure noise trader — they hold one third of the evidence. When their trades stop being credible, that bit is simply *deleted from the aggregate*, and prices revert to the posterior-without-their-signal (exactly 0.5 in the pivotal states). No amount of subsidy lets the other traders reconstruct information they never had. Hanson's argument implicitly assumes the manipulator's information is redundant with the rest of the market; in thin oversight markets it usually is not. **Manipulation-as-subsidy is a large-market/redundant-information result, and its failure mode in small markets is information exclusion, not price bias.**

Two corollaries worth stating plainly:

1. **Prices don't end up biased in the bribed direction — they end up uninformative.** Mean prices in the broken states sit at ≈ 0.50, not 0.9. A monitor watching for "suspiciously favourable prices" will see nothing; the corruption shows up as *lost resolution* in exactly the states where the bribed agent's private evidence was decisive.
2. **The briber never actually buys the decision.** Even at a bounty of 0.2 (≈ 170× the bribed trader's honest expected profit), P(A implemented) moves only to 0.47–0.54 — a coin flip, not a purchase. The equilibrium's resistance is real. What the briber *does* get, cheaply, is degradation: decisions become noisier in the pivotal states. An adversary whose goal is to blind the oversight layer, rather than steer it, gets excellent value for money — and note the bribed trader comes out ahead overall (bonus receipts exceed market losses by construction), so "the manipulator loses money" is only true of their market book, not their whole position.

### Where do decisions actually flip?

* **Galanis LMSR: sharp threshold at bonus ≈ 0.02–0.05**, i.e. ~1–3× the LMSR cost of a full adverse price push (0.016) and ~15–40× the seat's honest profit. Below threshold: literally zero effect (the equilibrium is unchanged to 4 decimal places at bonus 0.01). Above: full information exclusion, saturating immediately — bonus 0.1 and bonus 1.0 produce identical equilibria. Corruption is all-or-nothing.
* **Hanson conditional: same threshold region (0.02–0.05), non-monotonic below it.** At bonus 0.02 decision accuracy *improves* (0.752 → 0.803): this small game has multiple coexisting equilibria (baseline coordination is imperfect, acc 0.752 < 0.875), and a small bias acts as an equilibrium selector that happens to pick a better-coordinated one. Genuine degradation starts at 0.05 and saturates at 0.679.
* **MetaDAO CFMM: no safe region at all in this sweep** — bonus 0.005 (less than the bribed seat's own honest profit of 0.007) already costs 10 accuracy points, and the response is non-monotonic (partial recovery at 0.05–0.1, worst at 0.2). The CFMM's path-dependent pricing is what gives it the best baseline (0.875, exactly the full-information optimum — better than Hanson's LMSR version at 0.752) — early trades anchor the pool cheaply — but that same anchoring makes the early rounds cheap to poison. **Baseline aggregation quality and corruption resistance are different axes, and here they anti-correlate.**

### Design reading

For an oversight market for agent collectives, these results suggest: (i) measure corruption resistance as *the bounty at which decisions change*, in units of a seat's honest expected profit — here that number is ~15–40× for the LMSR mechanisms and < 1× for the CFMM at these liquidity scales; (ii) don't rely on manipulator losses as a defence — the losses are real but small (≈ 0.3–1.8% of notional) compared to bounties an outside principal would happily pay, and the bribed *agent* profits net; (iii) watch for information exclusion, not price bias: the observable symptom of equilibrium corruption is pivotal-state prices parking at the no-signal posterior.

## Limitations

* **Three traders, one signal each.** The information-exclusion result is at its most extreme here: bribing one trader deletes 1/3 of the evidence. With many partially-redundant traders, Hanson's subsidy mechanism has more room to work; this sweep establishes the thin-market end, not the general case.
* **Tabular discretisation.** 9 (Galanis) / 7 (Hanson, MetaDAO) price points, 3 rounds, one quote per trader. The 0.9/0.1 price floor is a grid artifact; the tie-goes-to-A rule interacts with the 0.5 grid point. The threshold *locations* (0.02–0.05 etc.) are specific to b = 0.01 / K = 0.001 liquidity and should be read as "a few multiples of the honest cost-to-move," not as absolute numbers.
* **CFR+ in 3-player general-sum games converges to a coarse correlated equilibrium**, not necessarily Nash; the multiple-equilibria effects (Hanson's non-monotonicity, MetaDAO's recovery at 0.05–0.1) are equilibrium *selection* phenomena and could differ under other solvers or refinements. NashConv is ≤ 6×10⁻⁴ on every row (usually ≤ 3×10⁻⁵), so each reported profile is at least a near-equilibrium.
* **Equilibrium ≠ behavior.** Real (human or LLM) traders do not play best responses — Galanis 2026 shows LLM traders degrade with structural complexity where the equilibrium does not, and Ouyang & Sui 2026 show LLM traders carry human behavioral biases. Behavioral traders may fail to discount the manipulator (worse than these results) or overreact (different failure). The behavioral-LLM version of this sweep is the planned companion experiment.
* The manipulator's bonus is common knowledge (built into the game the CFR solver sees). A *covert* manipulator — traders uncertain whether player 0 is bribed — is a strictly harder and more realistic threat model; see Phase 2c below for the entry-game version.

## Phase 2a: entry — is a bribed trader worse than no trader at all?

The sweeps above *replaced* an informed trader with a manipulator, so the manipulator's seat came bundled with a third of the evidence. The sharper design question for an oversight market is about **entry**: given an honest 2-trader market, does *admitting* a possibly-bribed third participant ever leave you worse off than never admitting them (and their information) at all?

Setup (Galanis LMSR, `t3s111y2`, b = 0.01, 9-point grid, CFR+ 300 iters, NashConv ≤ 2×10⁻⁵ every row; script `galanis-market/scripts/entry_sweep.py`, raw results `galanis-market/results/entry_sweep_{BASE-2,T1,T2,T3}.json`):

* **BASE-2** — 2 honest traders observing bits *b*, *c*. Bit *a* still exists in nature (X depends on it) but nobody observes it. This is "the entrant and their information never existed." Decision acc **0.750**, the no-bit-*a* ceiling: pivotal states park at the 0.5 posterior.
* **BASE-3** — 3 honest informed traders (= T2 at bonus 0): acc **1.000**.
* **T1** — BASE-2 + an entrant with **no private information** + price bounty (Hanson's pure noise trader).
* **T2** — BASE-2 + an entrant who **uniquely observes bit *a*** + bounty. The key comparison: can bribed entry take the market *below* BASE-2?
* **T3** — BASE-2 + an entrant with a **redundant** signal (bit *b*, same as incumbent 1) + bounty.

Because the decision rule reads the **final** price, the entrant's seat in the rotation matters; each treatment is solved with the entrant moving **first** (both honest traders can correct afterwards) and **last** (nobody corrects).

**Decision accuracy** (final-price rule; BASE-2 = 0.750, BASE-3 = 1.000):

| bounty | T1-first | T1-last | T2-first | T2-last | T3-first | T3-last |
|---|---|---|---|---|---|---|
| 0.00 | 0.750 | 0.750 | **1.000** | **1.000** | 0.750 | 0.750 |
| 0.02 | 0.750 | 0.500 | 0.999 | 0.500 | 0.750 | 0.500 |
| 0.05 | 0.750 | 0.500 | 0.796 | 0.500 | 0.750 | 0.500 |
| 0.20 | 0.750 | 0.500 | 0.750 | 0.500 | 0.750 | 0.500 |

Three clean answers:

1. **When the entrant moves first (or anywhere the honest traders can answer), entry never hurts.** T1-first and T3-first are pinned at exactly the BASE-2 equilibrium for every bounty — the market simply ignores the entrant (T1's uninformed entrant doesn't even trade; T3's redundant entrant is out-inferred by the incumbent holding the same bit). T2-first degrades with the bounty but **bottoms out exactly at BASE-2**: per-state final prices at bounty 0.2 are identical to BASE-2's (0.9/0.5/0.1 pattern), i.e. full information exclusion, "as if bit *a* never existed" — never worse. The phase-1 exclusion result is therefore also the *floor* for entry, provided the honest side gets the last word.
2. **A bribed last-mover breaks the floor — and information has nothing to do with it.** With the entrant in the last seat, accuracy collapses to 0.500 < 0.750 at every positive bounty, *identically for T1, T2 and T3* — an entrant with zero information (T1) does exactly as much damage as one holding the pivotal bit. The channel is mechanical: the decision rule reads a price nobody can correct. At bounty 0.02 the bribed last-mover pushes the price to exactly the 0.5 decision threshold in the down states (any further is not worth the LMSR cost); at 0.2 they slam every state to 0.9. The equilibrium "defence" of discounting the manipulator still works — informed traders' prices *before* the last move are exactly the honest posteriors — but the decision rule never sees them.
3. **The subsidy dies in the last seat too.** For first-seat entry, informed profits rise with the bounty as in phase 1 (BASE-2 informed total 0.0029 → 0.0072 at T2-first's transition point). For last-seat entry the informed traders' profit is **unchanged at 0.0029 for every bounty**: they never get to trade against the manipulator's push, so the manipulator's losses (−0.008 at bounty 0.2) flow entirely to the market maker. Hanson's manipulation-as-subsidy transfer requires someone to *take the other side*; a last-mover manipulator has no counterparty except the AMM.

**Verdict on the open question:** admitting an informed-but-bribable entrant is worse than their information never existing **only through the last-mover/decision-reading channel**, not through any information channel. Equivalently: the danger of entry is not *who they are* or *what they know* but *when they trade relative to when the decision rule reads the price*. That immediately suggests the fix tested next: read something no single last move can own.

## Phase 2b: TWAP decision rules

The entry sweep localises the worst vulnerability in *what the decision rule reads*: a final price that one bribed trade can own. The obvious fix, in the spirit of the Othman–Sandholm critique of decision rules, is to act on the **time-averaged price (TWAP)** across all trades. This changes payoffs — the manipulator's price-target bonus is also paid on the TWAP — so every treatment was re-solved from scratch (`entry_sweep.py --twap`; raw results `entry_sweep_{BASE-2_T1,T2,REPL_T3}_twap.json`; NashConv ≤ 2×10⁻⁵).

**Decision accuracy, final-price rule vs TWAP rule** (BASE-2 = 0.750 under both rules):

| bounty | T2-last final | T2-last TWAP | T1-last final | T1-last TWAP | T2-first final | T2-first TWAP | replacement final | replacement TWAP |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 1.000 | 1.000 | 0.750 | 0.750 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.02 | 0.500 | **1.000** | 0.500 | **0.750** | 0.999 | **0.875** | 0.999 | — |
| 0.05 | 0.500 | **0.875** | 0.500 | **0.750** | 0.796 | 0.750 | 0.796 | 0.750 |
| 0.20 | 0.500 | **0.750** | 0.500 | **0.750** | 0.750 | 0.750 | 0.750 | 0.750 |

(T1-first and T3-first under TWAP are 0.750 at every bounty, identical to the final rule — those entrants were already ignored. T3-last TWAP is 0.750 at every bounty — the redundant last-mover is defused exactly like T1-last. "Replacement" is the phase-1 configuration — 3 informed traders, seat 0 bribed — re-solved under TWAP at three bounty levels.)

Scoring the three hypotheses:

* **(b) TWAP kills the last-mover advantage — confirmed, completely.** The uninformed bribed last-mover (T1-last), who breaks the market to 0.500 under the final-price rule at any bounty, does *nothing* under TWAP: 0.750 at every bounty. One trade is now only a third of the statistic, and pushing the average across the 0.5 threshold from honest prices costs more in the LMSR than the bounty pays back. T2-last degrades gradually (1.000 → 0.875 → 0.750) instead of collapsing at 0.02, and — the key point — **bottoms out at the BASE-2 floor of 0.750, never below**. Under TWAP the entry-never-hurts property holds in *every* seat: the phase-2a information-exclusion floor becomes position-independent.
* **(a) TWAP raises corruption thresholds — true only for late movers.** For T2-last the visible-damage threshold moves from <0.02 to ~0.05 and the saturation damage shrinks from −0.50 to −0.25 accuracy. But for *early* manipulators TWAP is slightly **worse**: T2-first at bounty 0.02 drops to 0.875 under TWAP vs 0.999 under final, and the phase-1 replacement config loses its partial-recovery region (0.796 → 0.750 at bounty 0.05). The reason is the mirror image of the last-mover fix: a final-price rule lets honest traders *erase* an early manipulative push before the rule reads; a time-average *remembers* it at weight 1/3, and no later correction can remove it.
* **(c) TWAP costs no baseline accuracy — confirmed** at this discretisation: every honest configuration (BASE-2 and all bonus-0 rows) is identical under both rules. (With only 3 trades in the average, a finer grid or longer horizon would show a small cost from the uninformative early prices lingering in the statistic.)

**Verdict:** TWAP is a *reallocation of vulnerability from the end of the tape to the beginning* — and the trade is clearly favourable. The end of the tape is where the catastrophic, information-free attack lived (any bribed entrant → 0.500 < BASE-2); the beginning only admits bounded degradation that saturates at the exclusion floor (0.750). And early pollution is dilutable in expectation — add more honest trades after it and the manipulator's weight in the average falls — whereas final-price pollution is not correctable by anything. Design rule: the decision statistic's ownership should be spread over more trades than any one participant can make.

## Phase 2c: type uncertainty (T2u) — can a covert manipulator hide in the honest pool?

Everything above makes the bounty common knowledge. The realistic threat is covert: the market knows an entrant *might* be bribed, not whether this one is. T2u modifies T2 with a second chance node — with probability q the entrant is **bribed** (saturation bounty 0.2), with 1−q **honest** — drawn after ω and observed by the entrant alone. Solved at q ∈ {0.25, 0.5}, both seats, final-price rule, 300 CFR+ iterations (`t2u_sweep.py`; raw results `t2u_{first,last}_q0.25_0.5.json`; NashConv ≤ 2.4×10⁻⁵). The live question: does the bribed type *hide in the honest pool* and retain influence he loses under common knowledge — and does the resulting suspicion make honest traders' strategies less informative, degrading accuracy below the known-type mixture?

**Decision accuracy** (T2-known references: honest entrant 1.000; bribed-known 0.750 first seat / 0.500 last seat; BASE-2 = 0.750):

| config | acc \| honest type | acc \| bribed type | expected acc | known-type mixture (1−q)·1.0 + q·acc_known |
|---|---|---|---|---|
| T2u-first q=0.25 | 1.000 | 0.750 | **0.9375** | 0.9375 |
| T2u-first q=0.50 | 1.000 | 0.750 | **0.8750** | 0.8750 |
| T2u-last q=0.25 | 1.000 | 0.500 | **0.8750** | 0.8750 |
| T2u-last q=0.50 | 1.000 | 0.500 | **0.7500** | 0.7500 |

**Verdict on decisions: type uncertainty is exactly neutral.** In all four solved games, expected accuracy equals the known-type mixture to four decimals, and the conditional accuracies equal their common-knowledge values (honest 1.000, bribed 0.750/0.500). Uncertainty about *who* is corrupt neither amplifies the damage (no contagion from suspicion to honest states) nor dampens it (hiding buys the briber no extra decision flips). On the accuracy question, the skeptical position is vindicated.

**But the hiding effect is real — it lives in the prices, not the decisions.** Per-state mean prices for T2u-first (pivotal states; honest play would put b, c at 0.9 and f, g at 0.1):

| state (truth) | known honest | T2u q=0.25, honest type | T2u q=0.25, bribed type | q=0.5 honest | q=0.5 bribed | known bribed |
|---|---|---|---|---|---|---|
| b, c (X=1) | 0.90 | 0.79 | 0.80 | 0.68 | 0.70 | 0.50 |
| f, g (X=0) | 0.10 | 0.10 | 0.79 | 0.10 | 0.69 | 0.50 |

Two opposite distortions, cleanly visible:

* **The bribed type partially pools and keeps real price influence.** Under common knowledge he is fully discounted — pivotal prices sit at 0.50 whatever he does. Under q = 0.25 his up-pushes carry the wrong states f, g to ≈ 0.79 (q = 0.5: ≈ 0.69): the market can't tell his bribed push from an honest "my bit is 1," so it partially believes him. Hiding also makes manipulation cheaper: his market loss shrinks from −0.0022 (known) to −0.0014 (q = 0.25) while achieving *more* price movement.
* **The honest type pays for the suspicion.** His genuine "bit = 1" evidence now moves pivotal prices only to 0.79 / 0.68 instead of 0.90 — the mirror-image discount. (His profits are untouched: +0.00124, same as a trusted honest entrant; the discount costs resolution, not P&L.)

At this grid the two distortions cancel exactly at the 0.5 decision threshold — polluted wrong-state prices (0.69–0.79) and discounted right-state prices (0.68–0.79) both stay on the same side of 0.5 that full-information play would choose, so no decision flips beyond the mixture. That cancellation is not a theorem: a decision threshold at 0.7, or a finer price grid, would let the pooling shift bite. The honest reading of the price columns is that **covert manipulation converts the market from "blind in the bribed states" to "blurry in all pivotal states"**: log-resolution is degraded in every state where the entrant's bit matters, in proportion to q, even conditional on the entrant being honest. In the last seat, by contrast, hiding is worthless — the bribed type's final push (all states → 0.9) is separating but arrives after anyone could use the information, so all conditional statistics match the common-knowledge play exactly.

## Next steps

1. **Redundancy as a defence:** re-run with 4–5 traders holding overlapping/correlated signals (or one insider seat) to find the redundancy level at which Hanson's subsidy mechanism starts repairing prices instead of merely compensating traders — i.e., where does the small-market result hand over to Hanson's large-market intuition?
2. **Behavioral companion experiment:** replace CFR+ traders with LLM agents (Galanis-style harness) under identical bonus schedules, and compare the empirical corruption threshold and subsidy-capture share against these equilibrium curves. Divergence in either direction is informative: equilibrium play is the rationality ceiling, and the gap is the additional fragility an agent-collective oversight market would actually exhibit.
