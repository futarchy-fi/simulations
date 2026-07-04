# galanis-market

OpenSpiel formulation of the Galanis (2026) prediction-market game.

> Galanis, Spyros (2026). *Information Aggregation with AI Agents.* arXiv:2604.20050.
> [arXiv](https://arxiv.org/abs/2604.20050) · [RePEc](https://d.repec.org/n?u=RePEc:arx:papers:2604.20050)
>
> The empirical baselines in [`src/galanis_market/galanis_empirics.py`](src/galanis_market/galanis_empirics.py) and
> [`results/equilibria.md`](results/equilibria.md) ("paper mean/median LE") are the paper's LLM-agent results
> (Tables 5–6); our CFR+ equilibria provide the rational benchmark alongside them.

## Goal

Derive Bayes-Nash equilibria of the 4 information structures (Easy → Very Hard) using tabular CFR+, and compare the equilibrium price distributions to the empirical LLM behaviour reported in the paper.

## Quick start

```bash
pip install -e .
pytest -q
python scripts/solve_all.py
```

## Layout

```
src/galanis_market/
  lmsr.py         LMSR cost / price / payoff primitives
  structures.py   The 4 information & payoff structures
  game.py         pyspiel.Game subclass
  solve.py        CFR+ runner + exploitability reporting
tests/            Sanity tests
scripts/          End-to-end driver scripts
results/          Solved equilibria (gitignored)
```

## Action discretisation

LMSR allows continuous trades; CFR is tabular, so each trader picks from a discrete set of target prices on a grid `{0.05, 0.10, ..., 0.95}` (configurable). Cost of moving the price is computed exactly by the LMSR formula given the chosen target.
