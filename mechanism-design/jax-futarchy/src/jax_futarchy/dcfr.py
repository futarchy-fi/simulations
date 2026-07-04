"""Deep CFR via external sampling on JAX-native games.

External sampling: at the traverser's info-state we branch on every
action; at every other player's info-state we sample a single action
from their current regret-matching strategy. Counterfactual values are
the terminal payoffs along each branch; regrets are
``v_a - sum_a' pi(a') v_a'``.

We unroll the traversal for the fixed ``num_rounds`` so JAX can JIT it.
This implementation handles num_rounds in {3, 6, 9}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from jax_futarchy.game import GalanisGame, GalanisState
from jax_futarchy.networks import RegretNet, regret_matching


@dataclass
class DCFRConfig:
    iterations: int = 100
    traversals_per_iter: int = 1024
    train_steps_per_iter: int = 16
    train_batch: int = 1024
    hidden: int = 64
    depth: int = 2
    lr: float = 1e-3
    buffer_capacity: int = 100_000
    seed: int = 0


def make_dcfr(game: GalanisGame, config: DCFRConfig):
    """Build closures that run Deep CFR for the given game configuration."""
    net = RegretNet(num_actions=game.num_actions, hidden=config.hidden,
                     depth=config.depth)
    K = game.num_actions
    R = game.num_rounds

    def regret_strategy_for(params_p, info):
        regrets = net.apply(params_p, info)
        return regret_matching(regrets)

    def sample_action_for(params_p, info, key):
        strategy = regret_strategy_for(params_p, info)
        return jax.random.categorical(key, jnp.log(strategy + 1e-12))

    # ---- One-shot rollout: from a given state with all players sampling ----
    def rollout_from(state, params_all, key):
        """Sample actions for all remaining steps; return terminal trader_profits."""
        def body(carry, _):
            state, key = carry
            active = game.current_player(state)
            key, sk = jax.random.split(key)
            info = game.info_state(state, active)
            # We pick the active player's params using lax.switch.
            regrets = jax.lax.switch(
                active,
                [lambda p=p: net.apply(params_all[p], info) for p in range(3)],
            )
            action = jax.random.categorical(sk, jnp.log(regret_matching(regrets) + 1e-12))
            new_state = jax.lax.cond(state.finished, lambda s: s,
                                      lambda s: game.step(s, action), state)
            return (new_state, key), None
        (final, _), _ = jax.lax.scan(body, (state, key), None, length=R)
        return final.trader_profits

    # ---- External sampling: enumerate K actions at one node, sample elsewhere ----
    def value_after_action(state, action, params_all, key):
        """Apply `action`, then sample to terminal. Returns trader_profits."""
        next_state = game.step(state, action)
        return rollout_from(next_state, params_all, key)

    def regrets_at(state, traverser, params_all, key):
        """Compute regrets at this state if `traverser` is the active player."""
        info = game.info_state(state, traverser)
        own_strategy = regret_strategy_for(params_all[traverser], info)
        # Vmap over K actions.
        keys = jax.random.split(key, K)
        all_profits = jax.vmap(lambda a, k: value_after_action(state, a, params_all, k))(
            jnp.arange(K), keys
        )
        # Pick the traverser's column.
        action_values = all_profits[:, traverser]  # shape [K]
        v = jnp.sum(own_strategy * action_values)
        regrets = action_values - v
        return info, regrets, v

    def make_traverse(traverser: int):
        """Build a traversal closure with `traverser` baked in (static)."""

        def traverse(key, params_all):
            key, k_init = jax.random.split(key)
            state = game.init(k_init)

            info_dim = game.info_state_dim()
            max_visits = R // 3 + 1
            infos_buf = jnp.zeros((max_visits, info_dim), dtype=jnp.float32)
            regrets_buf = jnp.zeros((max_visits, K), dtype=jnp.float32)
            masks_buf = jnp.zeros((max_visits,), dtype=jnp.float32)

            def body(carry, _):
                state, key, infos, regrets, masks, visit_idx = carry
                active = game.current_player(state)
                is_traverser = active == traverser
                key, k_sample, k_regret = jax.random.split(key, 3)

                # Compute regrets at this state (assuming traverser is active).
                info_t = game.info_state(state, traverser)
                own_strategy = regret_strategy_for(params_all[traverser], info_t)
                keys = jax.random.split(k_regret, K)
                all_profits = jax.vmap(
                    lambda a, k: value_after_action(state, a, params_all, k)
                )(jnp.arange(K), keys)
                action_values = all_profits[:, traverser]
                v = jnp.sum(own_strategy * action_values)
                regrets_t = action_values - v

                # Active player's sampled action (works whether traverser or not).
                info_active = game.info_state(state, active)
                # Pick the active player's params via lax.switch (active is traced).
                regrets_active = jax.lax.switch(
                    active,
                    [lambda p=p: net.apply(params_all[p], info_active)
                     for p in range(3)],
                )
                strategy_active = regret_matching(regrets_active)
                action = jax.random.categorical(
                    k_sample, jnp.log(strategy_active + 1e-12)
                )

                new_state = game.step(state, action)

                # Record this visit iff traverser.
                new_infos = jnp.where(
                    is_traverser, infos.at[visit_idx].set(info_t), infos
                )
                new_regrets = jnp.where(
                    is_traverser, regrets.at[visit_idx].set(regrets_t), regrets
                )
                new_masks = jnp.where(
                    is_traverser, masks.at[visit_idx].set(1.0), masks
                )
                new_visit_idx = jnp.where(is_traverser, visit_idx + 1, visit_idx)

                return (new_state, key, new_infos, new_regrets, new_masks,
                        new_visit_idx), None

            carry = (state, key, infos_buf, regrets_buf, masks_buf,
                     jnp.array(0, dtype=jnp.int32))
            (_, _, infos, regrets, masks, _), _ = jax.lax.scan(
                body, carry, jnp.arange(R)
            )
            return infos, regrets, masks

        return jax.jit(jax.vmap(traverse, in_axes=(0, None)))

    traverse_v_per_player = [make_traverse(p) for p in range(3)]

    # ---- Trainer ----
    def init_trainer(key):
        keys = jax.random.split(key, 3)
        dummy = jnp.zeros(game.info_state_dim(), dtype=jnp.float32)
        params = [net.init(k, dummy) for k in keys]
        optim = optax.adam(config.lr)
        opt_states = [optim.init(p) for p in params]
        return params, opt_states, optim

    def loss_fn(params_p, infos, target_regrets):
        preds = jax.vmap(net.apply, in_axes=(None, 0))(params_p, infos)
        return jnp.mean((preds - target_regrets) ** 2)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def apply_update(params_p, opt_state, grads, optim_fn):
        # optim is wrapped in closure outside JIT; can't be a JIT arg easily.
        # We'll do this outside JIT instead.
        pass

    def train_loop(key, verbose=False):
        rng = jax.random.PRNGKey(config.seed)
        params, opt_states, optim = init_trainer(rng)
        # Per-player replay buffer of (info, accumulated_regret) samples.
        # We accumulate regrets across iterations (the CFR way): each
        # info-state's running sum of regrets is what the network should
        # predict. This makes the policy = regret-matching the average.
        buffers = [
            {"infos": jnp.zeros((0, game.info_state_dim()), dtype=jnp.float32),
             "regrets_cumsum": jnp.zeros((0, K), dtype=jnp.float32)}
            for _ in range(3)
        ]

        for it in range(config.iterations):
            rng, k_t = jax.random.split(rng)
            for p in range(3):
                keys = jax.random.split(jax.random.fold_in(k_t, p),
                                         config.traversals_per_iter)
                infos, regrets, masks = traverse_v_per_player[p](keys, params)
                infos_flat = infos.reshape(-1, infos.shape[-1])
                regrets_flat = regrets.reshape(-1, regrets.shape[-1])
                masks_flat = masks.reshape(-1)
                idx = jnp.where(masks_flat > 0)[0]
                new_infos = infos_flat[idx]
                new_regrets = regrets_flat[idx]
                # Append to buffer.
                buffers[p]["infos"] = jnp.concatenate(
                    [buffers[p]["infos"], new_infos]
                )
                buffers[p]["regrets_cumsum"] = jnp.concatenate(
                    [buffers[p]["regrets_cumsum"], new_regrets]
                )
                # Cap buffer size.
                cap = config.buffer_capacity
                if buffers[p]["infos"].shape[0] > cap:
                    # Reservoir-style: keep most recent.
                    buffers[p]["infos"] = buffers[p]["infos"][-cap:]
                    buffers[p]["regrets_cumsum"] = buffers[p]["regrets_cumsum"][-cap:]

                # Train on buffer.
                buf_infos = buffers[p]["infos"]
                buf_regrets = buffers[p]["regrets_cumsum"]
                n = buf_infos.shape[0]
                for s in range(config.train_steps_per_iter):
                    rng, k_b = jax.random.split(rng)
                    if n > config.train_batch:
                        idx_b = jax.random.randint(k_b, (config.train_batch,), 0, n)
                        bi = buf_infos[idx_b]
                        br = buf_regrets[idx_b]
                    else:
                        bi = buf_infos
                        br = buf_regrets
                    loss, grads = grad_fn(params[p], bi, br)
                    updates, opt_states[p] = optim.update(grads, opt_states[p])
                    params[p] = optax.apply_updates(params[p], updates)
            if verbose and (it % 10 == 0 or it == config.iterations - 1):
                print(f"  iter {it}/{config.iterations}  buf_size={[b['infos'].shape[0] for b in buffers]}", flush=True)
        return params

    return params_to_strategy_for_player, train_loop, net


def params_to_strategy_for_player(net, params_p, info):
    regrets = net.apply(params_p, info)
    return regret_matching(regrets)


def evaluate(game: GalanisGame, net: RegretNet, params, n_samples: int = 4000,
              seed: int = 0):
    """Monte Carlo estimate of mean log error under the learned policy."""
    import jax.random as jr
    K = game.num_actions
    rng = jr.PRNGKey(seed)
    # For each omega, simulate n_samples rollouts.
    mean_log_err_acc = 0.0
    median_log_err_acc = 0.0
    by_omega = {}
    eps = 1e-15
    for omega_idx in range(8):
        prices = []
        for s in range(n_samples):
            rng, k_init = jr.split(rng)
            # Init with specific omega.
            from jax_futarchy.game import SIGNAL_TABLE, GalanisState
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
                regrets = net.apply(params[active], info)
                strategy = regret_matching(regrets)
                rng, sk = jr.split(rng)
                action = int(jr.categorical(sk, jnp.log(strategy + 1e-12)))
                state = game.step(state, jnp.array(action))
            prices.append(float(state.price_history[-1]))
        prices.sort()
        mean_p = sum(prices) / len(prices)
        median_p = prices[len(prices) // 2]
        x = int(game.x_table[omega_idx])
        def _le(p):
            p = max(eps, min(1 - eps, p))
            import math
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


__all__ = ["DCFRConfig", "make_dcfr", "params_to_strategy_for_player", "evaluate"]
