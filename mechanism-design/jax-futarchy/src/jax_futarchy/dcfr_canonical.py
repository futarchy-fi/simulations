"""Canonical Deep CFR (Brown, Lerer, Gibson, Sandholm 2019).

Key components:
1. Two networks per player:
   - Advantage network V_θ: predicts cumulative regret per action
   - Strategy network π_φ: predicts time-averaged strategy
2. Two reservoir-sampled buffers per player:
   - M_V: (info_state, regrets, iter_t) tuples
   - M_π: (info_state, strategy, iter_t) tuples
3. Each iteration:
   a. For each player p (as traverser):
      - Sample K trajectories with external sampling
      - At p's nodes: compute regrets, store in M_V[p] (reservoir, weighted by t)
      - At p's nodes: also store the strategy in M_π[p]
   b. Retrain V_θ on M_V (from scratch each iter to avoid drift)
   c. Retrain π_φ on M_π (from scratch each iter)
4. Final policy: π_φ output (not V_θ regret-matching)

Compared to dcfr.py: this is the *actual* Brown-Lerer-Gibson-Sandholm
algorithm. The "dcfr.py" simpler variant uses only V_θ and reports its
regret-matching strategy as the policy, which is biased and noisier.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_futarchy.game import GalanisGame, GalanisState, SIGNAL_TABLE
from jax_futarchy.networks import RegretNet, regret_matching


@dataclass
class CanonicalDCFRConfig:
    iterations: int = 1000
    traversals_per_iter: int = 1024
    train_steps: int = 4000             # SGD steps per iter on each net
    train_batch: int = 1024
    hidden: int = 128
    depth: int = 3
    lr: float = 1e-3
    buffer_capacity: int = 500_000      # reservoir capacity per player per buffer
    retrain_from_scratch: bool = True   # re-init network params each iter
    seed: int = 0


class ReservoirBuffer:
    """Reservoir-sampled buffer storing (info_state, target, weight)."""

    def __init__(self, capacity: int, info_dim: int, target_dim: int, seed: int = 0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.infos = np.zeros((capacity, info_dim), dtype=np.float32)
        self.targets = np.zeros((capacity, target_dim), dtype=np.float32)
        self.weights = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.total_seen = 0

    def add_batch(self, infos: np.ndarray, targets: np.ndarray, weight: float):
        # Reservoir-add each sample with probability decreasing over time.
        n = infos.shape[0]
        for i in range(n):
            self.total_seen += 1
            if self.size < self.capacity:
                self.infos[self.size] = infos[i]
                self.targets[self.size] = targets[i]
                self.weights[self.size] = weight
                self.size += 1
            else:
                j = self.rng.integers(0, self.total_seen)
                if j < self.capacity:
                    self.infos[j] = infos[i]
                    self.targets[j] = targets[i]
                    self.weights[j] = weight

    def sample(self, batch_size: int, rng) -> tuple:
        n = min(self.size, batch_size)
        if self.size <= batch_size:
            idx = np.arange(self.size)
        else:
            # Weight-proportional sampling (iteration-weighted).
            p = self.weights[:self.size]
            p = p / p.sum()
            idx = self.rng.choice(self.size, size=batch_size, replace=True, p=p)
        return self.infos[idx], self.targets[idx]


def make_canonical_dcfr(game: GalanisGame, config: CanonicalDCFRConfig):
    """Build canonical Deep CFR trainer."""
    regret_net = RegretNet(
        num_actions=game.num_actions, hidden=config.hidden, depth=config.depth
    )
    # Strategy net produces logits over actions (softmax → strategy).
    strategy_net = RegretNet(
        num_actions=game.num_actions, hidden=config.hidden, depth=config.depth
    )
    K = game.num_actions
    R = game.num_rounds

    def regret_to_strategy(regrets):
        return regret_matching(regrets)

    # ---- External-sampling traversal (same shape as dcfr.py) ----
    def rollout_from(state, params_regret_all, key):
        def body(carry, _):
            state, key = carry
            active = game.current_player(state)
            key, sk = jax.random.split(key)
            info = game.info_state(state, active)
            regrets = jax.lax.switch(
                active,
                [lambda p=p: regret_net.apply(params_regret_all[p], info) for p in range(3)],
            )
            action = jax.random.categorical(sk, jnp.log(regret_matching(regrets) + 1e-12))
            new_state = jax.lax.cond(state.finished, lambda s: s,
                                      lambda s: game.step(s, action), state)
            return (new_state, key), None
        (final, _), _ = jax.lax.scan(body, (state, key), None, length=R)
        return final.trader_profits

    def value_after_action(state, action, params_regret_all, key):
        next_state = game.step(state, action)
        return rollout_from(next_state, params_regret_all, key)

    def make_traverse(traverser: int):
        def traverse(key, params_regret_all):
            key, k_init = jax.random.split(key)
            state = game.init(k_init)
            info_dim = game.info_state_dim()
            max_visits = R // 3 + 1
            infos = jnp.zeros((max_visits, info_dim), dtype=jnp.float32)
            regrets_buf = jnp.zeros((max_visits, K), dtype=jnp.float32)
            strategy_buf = jnp.zeros((max_visits, K), dtype=jnp.float32)
            masks = jnp.zeros((max_visits,), dtype=jnp.float32)

            def body(carry, _):
                state, key, infos, regrets, strategies, masks, visit_idx = carry
                active = game.current_player(state)
                is_traverser = active == traverser
                key, k_sample, k_regret = jax.random.split(key, 3)

                # Compute regrets and current strategy at this state.
                info_t = game.info_state(state, traverser)
                own_regrets = regret_net.apply(params_regret_all[traverser], info_t)
                own_strategy = regret_matching(own_regrets)
                keys = jax.random.split(k_regret, K)
                all_profits = jax.vmap(
                    lambda a, k: value_after_action(state, a, params_regret_all, k)
                )(jnp.arange(K), keys)
                action_values = all_profits[:, traverser]
                v = jnp.sum(own_strategy * action_values)
                regrets_t = action_values - v

                # Action actually taken (current strategy of active player).
                info_active = game.info_state(state, active)
                regrets_active = jax.lax.switch(
                    active,
                    [lambda p=p: regret_net.apply(params_regret_all[p], info_active)
                     for p in range(3)],
                )
                strategy_active = regret_matching(regrets_active)
                action = jax.random.categorical(
                    k_sample, jnp.log(strategy_active + 1e-12)
                )
                new_state = game.step(state, action)

                new_infos = jnp.where(is_traverser, infos.at[visit_idx].set(info_t), infos)
                new_regrets = jnp.where(is_traverser, regrets.at[visit_idx].set(regrets_t), regrets)
                new_strategies = jnp.where(is_traverser, strategies.at[visit_idx].set(own_strategy), strategies)
                new_masks = jnp.where(is_traverser, masks.at[visit_idx].set(1.0), masks)
                new_visit_idx = jnp.where(is_traverser, visit_idx + 1, visit_idx)
                return (new_state, key, new_infos, new_regrets, new_strategies,
                        new_masks, new_visit_idx), None

            carry = (state, key, infos, regrets_buf, strategy_buf, masks,
                     jnp.array(0, dtype=jnp.int32))
            (_, _, infos, regrets, strategies, masks, _), _ = jax.lax.scan(
                body, carry, jnp.arange(R)
            )
            return infos, regrets, strategies, masks
        return jax.jit(jax.vmap(traverse, in_axes=(0, None)))

    traverse_v_per_player = [make_traverse(p) for p in range(3)]

    # ---- Network init + training ----
    def init_networks(key):
        keys_r = jax.random.split(key, 3)
        keys_s = jax.random.split(jax.random.fold_in(key, 1), 3)
        dummy = jnp.zeros(game.info_state_dim(), dtype=jnp.float32)
        regret_params = [regret_net.init(k, dummy) for k in keys_r]
        strategy_params = [strategy_net.init(k, dummy) for k in keys_s]
        return regret_params, strategy_params

    def regret_loss(params, infos, targets):
        preds = jax.vmap(regret_net.apply, in_axes=(None, 0))(params, infos)
        return jnp.mean((preds - targets) ** 2)

    def strategy_loss(params, infos, targets):
        # Strategy net outputs logits; targets are strategy distributions.
        logits = jax.vmap(strategy_net.apply, in_axes=(None, 0))(params, infos)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.sum(targets * log_probs, axis=-1))

    regret_grad = jax.value_and_grad(regret_loss)
    strategy_grad = jax.value_and_grad(strategy_loss)

    # Pre-JIT a multi-step SGD scanner that runs S SGD steps over a batched
    # set of (infos, targets) inside a single JIT call. This eliminates
    # Python dispatch overhead per step.
    def make_multistep(loss_grad_fn, optimizer):
        def multistep(params, opt_state, batches_infos, batches_targets):
            def step(carry, batch):
                params, opt_state = carry
                infos, targets = batch
                _, grads = loss_grad_fn(params, infos, targets)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), None
            (params, opt_state), _ = jax.lax.scan(
                step, (params, opt_state), (batches_infos, batches_targets)
            )
            return params, opt_state
        return jax.jit(multistep)

    optim = optax.adam(config.lr)
    regret_multistep = make_multistep(regret_grad, optim)
    strategy_multistep = make_multistep(strategy_grad, optim)

    def train_loop(key, verbose=False):
        rng_jax = jax.random.PRNGKey(config.seed)
        regret_params, strategy_params = init_networks(rng_jax)

        # Per-player buffers.
        regret_buffers = [
            ReservoirBuffer(config.buffer_capacity, game.info_state_dim(), K, seed=p)
            for p in range(3)
        ]
        strategy_buffers = [
            ReservoirBuffer(config.buffer_capacity, game.info_state_dim(), K, seed=100 + p)
            for p in range(3)
        ]

        # We retrain from scratch each iteration if config says so.

        for it in range(1, config.iterations + 1):
            rng_jax, k_t = jax.random.split(rng_jax)
            # Traversal phase: sample for each player as traverser.
            for p in range(3):
                keys = jax.random.split(jax.random.fold_in(k_t, p),
                                         config.traversals_per_iter)
                infos, regrets, strategies, masks = traverse_v_per_player[p](
                    keys, regret_params
                )
                infos_flat = np.asarray(infos.reshape(-1, infos.shape[-1]))
                regrets_flat = np.asarray(regrets.reshape(-1, regrets.shape[-1]))
                strats_flat = np.asarray(strategies.reshape(-1, strategies.shape[-1]))
                masks_flat = np.asarray(masks.reshape(-1))
                idx = np.where(masks_flat > 0)[0]
                regret_buffers[p].add_batch(
                    infos_flat[idx], regrets_flat[idx], weight=float(it)
                )
                strategy_buffers[p].add_batch(
                    infos_flat[idx], strats_flat[idx], weight=float(it)
                )

            # Train phase: optionally re-init from scratch.
            if config.retrain_from_scratch:
                regret_params, strategy_params = init_networks(
                    jax.random.fold_in(rng_jax, it * 2 + 7)
                )

            # Pre-sample all batches for this iter, train via JIT scan.
            def _stack_batches(buffer, n_steps, batch_size):
                infos_list = []
                targets_list = []
                for _ in range(n_steps):
                    bi, br = buffer.sample(batch_size, None)
                    infos_list.append(bi)
                    targets_list.append(br)
                return (jnp.asarray(np.stack(infos_list)),
                        jnp.asarray(np.stack(targets_list)))

            # Train regret nets via JIT-scanned SGD.
            for p in range(3):
                opt_state = optim.init(regret_params[p])
                bi_batched, br_batched = _stack_batches(
                    regret_buffers[p], config.train_steps, config.train_batch
                )
                regret_params[p], _ = regret_multistep(
                    regret_params[p], opt_state, bi_batched, br_batched
                )

            # Train strategy nets via JIT-scanned SGD.
            for p in range(3):
                opt_state = optim.init(strategy_params[p])
                bi_batched, bs_batched = _stack_batches(
                    strategy_buffers[p], config.train_steps, config.train_batch
                )
                strategy_params[p], _ = strategy_multistep(
                    strategy_params[p], opt_state, bi_batched, bs_batched
                )

            if verbose and (it == 1 or it % 50 == 0 or it == config.iterations):
                sizes = [b.size for b in regret_buffers]
                print(f"  iter {it}/{config.iterations}  buffer={sizes}", flush=True)
        return regret_params, strategy_params

    return train_loop, regret_net, strategy_net


def evaluate_canonical(
    game: GalanisGame,
    strategy_net,
    strategy_params,
    n_samples: int = 4000,
    seed: int = 0,
):
    """Evaluate using the *strategy* network (canonical: π_φ, not V_θ regret matching)."""
    import math
    import jax.random as jr
    rng = jr.PRNGKey(seed)
    eps = 1e-15
    mean_log_err_acc = 0.0
    median_log_err_acc = 0.0
    by_omega = {}
    for omega_idx in range(8):
        prices = []
        for s in range(n_samples):
            rng, k_init = jr.split(rng)
            state = GalanisState(
                omega=jnp.array(omega_idx, dtype=jnp.int32),
                signals=SIGNAL_TABLE[omega_idx],
                price_history=jnp.zeros(game.num_rounds + 1).at[0].set(game.initial_price),
                action_history=jnp.zeros(game.num_rounds, dtype=jnp.int32),
                cur_step=jnp.array(0, dtype=jnp.int32),
                trader_profits=jnp.zeros(3, dtype=jnp.float32),
                finished=jnp.array(False),
            )
            for _ in range(game.num_rounds):
                active = int(state.cur_step) % 3
                info = game.info_state(state, jnp.array(active))
                logits = strategy_net.apply(strategy_params[active], info)
                rng, sk = jr.split(rng)
                action = int(jr.categorical(sk, logits))
                state = game.step(state, jnp.array(action))
            prices.append(float(state.price_history[-1]))
        prices.sort()
        mean_p = sum(prices) / len(prices)
        median_p = prices[len(prices) // 2]
        x = int(game.x_table[omega_idx])

        def _le(p):
            p = max(eps, min(1 - eps, p))
            return -(x * math.log(p) + (1 - x) * math.log(1 - p))

        mean_log_err_acc += _le(mean_p)
        median_log_err_acc += _le(median_p)
        by_omega[chr(ord('a') + omega_idx)] = {
            "x": x, "mean_price": mean_p, "median_price": median_p,
            "log_error": _le(mean_p),
        }
    return {
        "mean_log_error": mean_log_err_acc / 8,
        "median_log_error": median_log_err_acc / 8,
        "by_omega": by_omega,
    }


__all__ = [
    "CanonicalDCFRConfig",
    "make_canonical_dcfr",
    "evaluate_canonical",
    "ReservoirBuffer",
]
