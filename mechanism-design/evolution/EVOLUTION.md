# EVOLUTION — capital-selection on prompt-genome trader populations

*Design spec, 2026-07-04. Status: designed, not yet implemented. Owner rationale: the
simulation corpus (MANIPULATION.md, KYLE.md, BATCH.md, results/llm-decision-market/)
validated the market mechanism but never tested the program's core bets — that
selection pressure (capital, evolution) produces trader populations whose markets
aggregate well and resist manipulation. Every simulated population so far was static;
the load-bearing qualification on Hanson's manipulation-as-subsidy argument (frozen-β
traders kill the accuracy subsidy, KYLE.md Q6) is a statement about populations that
cannot adapt. This program tests whether selection restores adaptation.*

## The loop

Population of N LLM trader agents (cheap models — Haiku-class). Genome = the agent's
system prompt (freeform, capped ~500 tokens). Each generation: agents trade M synthetic
markets with known ground truth on our own batch-cleared LMSR venue
(`mechanism-design/batch-amm` engine, per the final mechanism spec: damped competitive
sizing, R≥2 rounds, price-only disclosure, outcome settlement). Fitness = **log-wealth**
(Kelly-consistent; punishes blowups). Selection + LLM-driven mutation produce the next
generation. Everything seeded, logged, resumable.

Mutation operator: GEPA-style reflective rewriting — the mutator LLM sees the parent
prompt, its PnL attribution, and (Lamarckian variant) its trading transcript, and
proposes a rewrite. Blind-mutation variant kept as an ablation (Lamarckian converges
faster but is more exploit-prone; E2 is the check).

## Experiments (each gates the next)

### E0 — Heritability check (gate for everything)
~24 hand-written archetype prompts (careful Bayesian, aggressive Kelly-sizer, momentum,
contrarian, noise-fader, …) run as a static population. Measure between-prompt PnL
variance vs within-prompt (luck) variance. If prompts don't differ, selection has no
raw material and the program stops. Byproduct: the power calculation — markets-per-
generation M needed for selection signal > luck (prior guess M ≈ 50–100).

### E1 — Does selection work?
Full loop, ~20 generations. **Controls that make it science:**
- *mutation-without-selection* (drift) — distinguishes evolution from prompt lottery;
- *selection-without-mutation* — just picking the best static prompt; evolution must
  beat it to earn the name.
Headline metric is NOT agent wealth but **venue price accuracy (log loss) per
generation** — whether the market aggregates better as wealth concentrates in better
forecasters. Cost estimate: 24 agents × 60 markets × 20 gens × ~2 calls ≈ 60k
Haiku calls ≈ $100/run.

### E2 — Skill or exploit? (pre-registered before E1 runs)
Transfer evolved champions to a fresh market family and a differently-parameterized
venue. PnL persists → forecasting skill; collapses → venue exploits. **Pre-registered
stance: both outcomes count, as different results** — exploit-emergence is the
manipulation thesis arising by selection (nobody prompted it; feeds the oversight
essay), skill-transfer feeds the aggregation claim. Instrument for known exploits from
the corpus: LMSR curvature games, jamming (BATCH.md §9), oscillation harvesting.

### E3 — Does the market itself do the selecting? (the "real money" bet)
Replace tournament selection with the market's native mechanism: capital carries over
across generations, position sizing scales with wealth, bankruptcy = death, wealth
threshold = reproduction. Compare venue accuracy under wealth-selection vs designed
tournament. If the ledger matches the breeder, markets don't need one.

### E4 — Does selection restore the Hanson subsidy? (the crown jewel)
Add a bounty-carrying manipulator (per MANIPULATION.md / KYLE.md threat models) to the
evolving population. KYLE.md Q6 showed static LLM-like (frozen-β) traders collapse the
noise-flow accuracy subsidy (corr 0.77→0.19). Question: do evolved populations
re-learn to scale up against manipulative flow? If yes, manipulation-as-subsidy works
for *evolved* agent markets even though it fails for static ones — the most
Hanson-relevant result available to this program. Answers the shelved σ_u
stake-response question as a byproduct.

## Framework decision (survey 2026-07-04, verified via GitHub)

**Roll our own loop; steal parts.** Every surveyed framework (OpenEvolve ~6.6k★,
ShinkaEvolve ~1.3k★, GEPA ~5.5k★, darwinian_evolver, EvoAgentX, DSPy optimizers)
assumes fitness = independent on-demand re-evaluation of a candidate. Our fitness is a
**shared-environment** signal — agents' PnL interacts through the market, and that
interaction (E3, E4) *is the experiment*. Frameworks would silently violate it.
The loop itself is a few hundred lines against infrastructure we already have
(batch-amm engine + the `claude -p` harness from results/llm-decision-market/).

Steal: GEPA's reflective-mutation pattern (MIT); darwinian_evolver's
fitness-proportional sampling + novelty bonus (design reference only — AGPL);
OpenEvolve's island/MAP-Elites diversity ideas (e.g. bin agents by strategy style) if
E1 shows premature convergence. OpenEvolve/ShinkaEvolve remain the right tools if we
later want offline backtest-fitness evolution, where candidates really are independent.

## Prior art / novelty positioning

- **ATLAS (atlas-gic, ~2k★, General Intelligence Capital)** — closest prior art:
  25→31 trading agents, prompts-as-weights, market-outcomes-as-loss, worst-by-rolling-
  Sharpe gets its prompt rewritten. So "evolve trader prompts on market feedback"
  **already exists**. Our differentiation: (1) prediction/decision markets with known
  ground truth, not price trading — we can measure *aggregation quality*, they can only
  measure Sharpe; (2) venue accuracy as the headline metric (the market-level claim);
  (3) adversarial robustness under selection (E4 — no analog there); (4) controls
  (drift / selection-only) — ATLAS is a results release, not a controlled experiment.
  Read their methodology before building; do not build on the repo (prompts
  proprietary, replication scaffold only).
- **MiroShark (~1.4k★, AGPL)** — simulated Polymarket with hundreds of LLM personas, no
  evolution loop. We have a better-instrumented venue of our own; skim for persona
  diversity ideas only.
- **Darwin Gödel Machine (jennyzzt/dgm, stale)** — archive-based open-ended selection
  ideas if E1 convergence stalls.

## Fixed design choices

Synthetic markets with known truth first (unlimited supply, instant resolution); real
questions later. Haiku population, Haiku-or-Sonnet mutator. Elitism: top 2 unchanged.
Keep-top-half tournament in E1 (wealth dynamics deferred to E3 by design). Standing
pseudo-anonymity rule: price-only disclosure, no per-trader attribution to agents.

## Open choices (owner input welcome)

- Lamarckian (mutator sees transcript) vs blind mutation as the *default* — plan: run
  Lamarckian first, let E2 catch the damage.
- Population size / M / generations beyond the E0 power calc.
- Whether E3 wealth carry-over uses hard bankruptcy or a wealth floor.
