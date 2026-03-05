# Simulations

Mechanism design research and simulations for Futarchy.

## proposal-evaluation

A framework for studying how agents with private signals can collectively evaluate proposals. Each proposal has an unobservable quality and a public importance. Agents stake money to express their beliefs, and a mechanism aggregates these into approve/reject decisions.

The model separates the **environment** (proposals, agents, signals, utilities) from the **mechanism** (rules of the game), enabling systematic search over both agent strategies and mechanism designs.

See [proposal-evaluation/MODEL.md](proposal-evaluation/MODEL.md) for the formal specification.
