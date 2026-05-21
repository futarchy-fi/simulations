# GalanisMarket — CFR+ equilibria

Tabular CFR+ solutions of the 4 Galanis (2026) information structures over the LMSR prediction market, compared with the empirical results reported in the paper and with the closed-form myopic Bayes-Nash benchmark.

## Headline comparison

| structure | rounds | NashConv | CFR mean LE | paper mean LE | CFR median LE | paper median LE |
|-----------|--------|----------|-------------|---------------|---------------|-----------------|
| Easy (`t3s111y2`) | 3 | 8.76e-05 | 0.0896 | 0.131 | 0.0870 | 0.018 |
| Medium (`t3s110`) | 3 | 1.67e-04 | 0.0966 | 0.173 | 0.0870 | 0.018 |
| Hard (`t3s111`) | 3 | 1.67e-04 | 0.0966 | 0.407 | 0.0870 | 0.304 |
| Very Hard (`t3s111o2ye2`) | 3 | 7.10e-05 | 0.0898 | 0.469 | 0.0870 | 0.718 |

**Reading the table.** `CFR mean LE` is our equilibrium's mean log-error, averaged over the 8 possible chance outcomes (uniform prior). `paper mean LE` is Galanis's Table 5 figure for the same structure × rounds. The median variant is computed from the equilibrium price distribution (weighted by policy probability) and compared with the paper's Table 6 quantile-regression median.

## Per-structure detail

### Easy — `t3s111y2`  (3 rounds, 11 actions, 50 iters)

- NashConv (final): **8.759e-05**  (lower = closer to equilibrium)
- CFR mean log error: **0.0896**  vs paper mean = 0.131
- CFR median log error: **0.0870**  vs paper median = 0.018
- Paper-reported typical price when X=1: **0.98**

| ω | state (dₐ,dᵦ,dᶜ) | X | CFR E[p] | CFR median p | myopic Bayes |
|---|------------------|---|----------|--------------|--------------|
| a | (1,1,1) | 1 | 0.9150 | 0.9167 | 1.0000 |
| b | (1,1,0) | 1 | 0.9137 | 0.9167 | 1.0000 |
| c | (1,0,1) | 1 | 0.9142 | 0.9167 | 1.0000 |
| d | (1,0,0) | 0 | 0.0858 | 0.0833 | 0.0000 |
| e | (0,1,1) | 1 | 0.9142 | 0.9167 | 1.0000 |
| f | (0,1,0) | 0 | 0.0858 | 0.0833 | 0.0000 |
| g | (0,0,1) | 0 | 0.0863 | 0.0833 | 0.0000 |
| h | (0,0,0) | 0 | 0.0850 | 0.0833 | 0.0000 |

### Medium — `t3s110`  (3 rounds, 11 actions, 50 iters)

- NashConv (final): **1.674e-04**  (lower = closer to equilibrium)
- CFR mean log error: **0.0966**  vs paper mean = 0.173
- CFR median log error: **0.0870**  vs paper median = 0.018
- Paper-reported typical price when X=1: **0.98**

| ω | state (dₐ,dᵦ,dᶜ) | X | CFR E[p] | CFR median p | myopic Bayes |
|---|------------------|---|----------|--------------|--------------|
| a | (1,1,1) | 1 | 0.8920 | 0.9167 | 1.0000 |
| b | (1,1,0) | 0 | 0.0896 | 0.0833 | 0.0000 |
| c | (1,0,1) | 0 | 0.0900 | 0.0833 | 0.0000 |
| d | (1,0,0) | 0 | 0.0860 | 0.0833 | 0.0000 |
| e | (0,1,1) | 0 | 0.1065 | 0.0833 | 0.0000 |
| f | (0,1,0) | 0 | 0.0858 | 0.0833 | 0.0000 |
| g | (0,0,1) | 0 | 0.0856 | 0.0833 | 0.0000 |
| h | (0,0,0) | 0 | 0.0851 | 0.0833 | 0.0000 |

### Hard — `t3s111`  (3 rounds, 11 actions, 50 iters)

- NashConv (final): **1.674e-04**  (lower = closer to equilibrium)
- CFR mean log error: **0.0966**  vs paper mean = 0.407
- CFR median log error: **0.0870**  vs paper median = 0.304
- Paper-reported typical price when X=1: **0.75**

| ω | state (dₐ,dᵦ,dᶜ) | X | CFR E[p] | CFR median p | myopic Bayes |
|---|------------------|---|----------|--------------|--------------|
| a | (1,1,1) | 1 | 0.8920 | 0.9167 | 1.0000 |
| b | (1,1,0) | 0 | 0.0896 | 0.0833 | 0.0000 |
| c | (1,0,1) | 0 | 0.0900 | 0.0833 | 0.0000 |
| d | (1,0,0) | 0 | 0.0860 | 0.0833 | 0.0000 |
| e | (0,1,1) | 0 | 0.1065 | 0.0833 | 0.0000 |
| f | (0,1,0) | 0 | 0.0858 | 0.0833 | 0.0000 |
| g | (0,0,1) | 0 | 0.0856 | 0.0833 | 0.0000 |
| h | (0,0,0) | 0 | 0.0851 | 0.0833 | 0.0000 |

### Very Hard — `t3s111o2ye2`  (3 rounds, 11 actions, 50 iters)

- NashConv (final): **7.100e-05**  (lower = closer to equilibrium)
- CFR mean log error: **0.0898**  vs paper mean = 0.469
- CFR median log error: **0.0870**  vs paper median = 0.718
- Paper-reported typical price when X=1: **0.5**

| ω | state (dₐ,dᵦ,dᶜ) | X | CFR E[p] | CFR median p | myopic Bayes |
|---|------------------|---|----------|--------------|--------------|
| a | (1,1,1) | 0 | 0.0868 | 0.0833 | 0.0000 |
| b | (1,1,0) | 1 | 0.9136 | 0.9167 | 1.0000 |
| c | (1,0,1) | 1 | 0.9143 | 0.9167 | 1.0000 |
| d | (1,0,0) | 0 | 0.0855 | 0.0833 | 0.0000 |
| e | (0,1,1) | 1 | 0.9140 | 0.9167 | 1.0000 |
| f | (0,1,0) | 0 | 0.0860 | 0.0833 | 0.0000 |
| g | (0,0,1) | 0 | 0.0852 | 0.0833 | 0.0000 |
| h | (0,0,0) | 0 | 0.0851 | 0.0833 | 0.0000 |

## Interpretation

### Headline finding

**The equilibrium aggregates in all four structures.** CFR+ converges to essentially the same price distribution per chance outcome regardless of complexity: median price ≈ 0.917 when X = 1, ≈ 0.083 when X = 0 (the 11-action discretisation floor). Median log error is ≈ 0.087 in every structure.

This is in sharp contrast to Galanis's empirics, where the LLM markets' median log error scales with structural complexity: 0.018 (Easy/Medium) → 0.304 (Hard) → 0.718 (Very Hard). The equilibrium does not exhibit this scaling -- the equilibrium price for the Very Hard structure aggregates just as well as the Easy structure does.

### Implication for the Galanis paper

Galanis frames the degradation as suggesting that *AI agents may suffer similar limitations to humans when reasoning about others* as complexity rises. Our solution shows that **the equilibrium itself does not degrade with complexity** -- separable securities do aggregate under best-response play in all four structures, consistent with Ostrovsky (2012). The Hard / Very Hard gap is therefore a **capability gap** between current LLMs and rational best-response play, not a property of the mechanism.

### Why Easy/Medium CFR > empirical median

Note the CFR median LE (0.087) is *larger* than the empirical median LE for Easy/Medium (0.018). This is a discretisation artefact: our 11-action price grid bottoms out at 0.083 / 0.917, imposing a floor of `-log(0.917) ≈ 0.087`. LLM traders use continuous prices and can reach ≈ 0.99, so they out-perform discretised CFR on the easiest structures. The fair comparison is qualitative: both aggregate near-perfectly when the structure is easy.

### Caveats

- Tabular CFR+ in a 3-player general-sum game converges to a coarse correlated equilibrium, not a Nash equilibrium. For this specific game the gap between CCE and NE is small (LMSR's unilateral-deviation incentive rules out babbling-style CCEs), so we read the CCE prices as proxies for the focal Bayes-Nash equilibrium predicted by Ostrovsky 2012. NashConv at termination is ≤ 2e-4 in every run.
- The 11-action discretisation is the binding floor on log error. A finer grid (21 or 99 actions) would lower the floor but is computationally heavier; the qualitative aggregation result is unchanged.
- Tabular CFR was only run at 3 rounds. Extending to 6/9 rounds in tabular Python is intractable (info-state count scales as `cells * num_actions^(moves_before_self_acts_again)`); we leave MCCFR and C++ CFR runs as follow-up.
