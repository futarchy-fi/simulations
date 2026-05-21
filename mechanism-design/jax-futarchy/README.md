# jax-futarchy

JAX-native re-implementation of the futarchy games with **Deep CFR** for large game spaces. Designed for the 9-round Galanis game (and beyond), where tabular CFR / Python MCCFR are intractable due to Python state-clone overhead.

## What's here

- `src/jax_futarchy/game.py` — Galanis market as pure-functional JAX. State is a `NamedTuple` of arrays; `step` is JIT-able and vmap-able.
- `src/jax_futarchy/lmsr.py` — LMSR primitives in JAX (logit, cost, shares, payoff).
- `src/jax_futarchy/networks.py` — Flax `RegretNet` (MLP) + regret-matching.
- `src/jax_futarchy/dcfr.py` — External-sampling Deep CFR with per-player replay buffers.

## Performance

JAX rollouts on this 9-round game: **~250,000 games/sec** on a single CPU core (CPU-only JAX 0.4.30). With a GPU (e.g. DGX Spark with CUDA), expect another 10–100× lift.

Compare with pure-Python OpenSpiel MCCFR on the same game: ~22 iterations/sec, i.e. ~660 game-steps/sec. JAX is roughly **1000× faster** for the env stepping, before training overhead.

## Status

| Component | Status |
|---|---|
| JAX Galanis game | working (3, 6, 9 rounds; configurable num_actions) |
| Deep CFR trainer | working but convergence quality is provisional |
| 3-round Easy DCFR (5 actions, 100 iter buffer) | mean LE 0.39 vs CFR+ baseline 0.18 |
| 9-round Easy DCFR | running |
| Hanson / MetaDAO in JAX | not yet ported |
| Manipulator support in JAX game | not yet ported |

## Known limitations of the current Deep CFR implementation

- We track a per-player **regret network** but no separate **strategy network**. The reported policy is the final-iteration's regret-matched strategy, not the time-averaged strategy a standard Deep CFR would produce. Convergence to NE is therefore only approximate.
- The replay buffer is FIFO-capped (50K–100K samples) rather than reservoir-sampled.
- Network is a small MLP (64 hidden × 2 layers). Larger nets may help on harder structures.
- We do not separately handle the chance node — `init` draws ω each traversal, so each traversal sees one ω. This is correct but high-variance compared with averaging across all 8 ω.

## Reproducing

```bash
pip install -e .
python3 -c "
import jax
from jax_futarchy.game import GalanisGame
from jax_futarchy.dcfr import make_dcfr, DCFRConfig, evaluate

game = GalanisGame(structure='t3s111y2', num_rounds=3, num_actions=5)
config = DCFRConfig(iterations=100, traversals_per_iter=512, train_batch=512)
_, train_loop, net = make_dcfr(game, config)
params = train_loop(jax.random.PRNGKey(0), verbose=True)
result = evaluate(game, net, params, n_samples=2000)
print('mean LE:', result['mean_log_error'])
"
```
