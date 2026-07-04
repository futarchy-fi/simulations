"""Kyle-style simultaneous batch decision markets: corruption theory.

Modules:
    closed_forms  -- baseline N-trader linear equilibrium closed forms (+ SymPy derivation)
    decision      -- logistic implementation rule q(p), Gaussian expectation helpers
    onebatch      -- one-batch linear-Gaussian equilibrium machinery (baseline, corruption, entry)
    mc            -- Monte Carlo verification + unilateral-deviation tests
    twap          -- T-batch myopic-lambda dynamics, TWAP vs last-batch decision statistic
    arrival       -- Q7: in-window information arrival (public stream / staggered signals)
    amm           -- fixed-impact (linearised LMSR) AMM variant
"""
