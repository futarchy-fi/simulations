"""Hanson-specific Deep CFR. Mirrors dcfr.py but calls game.terminal_profits()
at the end of each rollout (Hanson holdings need post-game resolution).
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from jax_futarchy.hanson_game import HansonGame, HansonState
from jax_futarchy.networks import RegretNet, regret_matching


@dataclass
class HansonDCFRConfig:
    iterations: int = 300
    traversals_per_iter: int = 512
    train_steps_per_iter: int = 16
    train_batch: int = 512
    hidden: int = 64
    depth: int = 2
    lr: float = 1e-3
    buffer_capacity: int = 100_000
    seed: int = 0


def make_dcfr(game: HansonGame, config: HansonDCFRConfig):
    net = RegretNet(num_actions=game.num_combined_actions,
                     hidden=config.hidden, depth=config.depth)
    K = game.num_combined_actions
    R = game.num_rounds

    def regret_strategy_for(params_p, info):
        regrets = net.apply(params_p, info)
        return regret_matching(regrets)

    def rollout_from(state, params_all, key):
        def body(carry, _):
            state, key = carry
            active = game.current_player(state)
            key, sk = jax.random.split(key)
            info = game.info_state(state, active)
            regrets = jax.lax.switch(
                active,
                [lambda p=p: net.apply(params_all[p], info) for p in range(3)],
            )
            action = jax.random.categorical(sk, jnp.log(regret_matching(regrets) + 1e-12))
            new_state = jax.lax.cond(state.finished, lambda s: s,
                                      lambda s: game.step(s, action), state)
            return (new_state, key), None
        (final, _), _ = jax.lax.scan(body, (state, key), None, length=R)
        return game.terminal_profits(final)  # [3]

    def value_after_action(state, action, params_all, key):
        next_state = game.step(state, action)
        return rollout_from(next_state, params_all, key)

    def make_traverse(traverser: int):
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
                info_t = game.info_state(state, traverser)
                own_strategy = regret_strategy_for(params_all[traverser], info_t)
                keys = jax.random.split(k_regret, K)
                all_profits = jax.vmap(
                    lambda a, k: value_after_action(state, a, params_all, k)
                )(jnp.arange(K), keys)
                action_values = all_profits[:, traverser]
                v = jnp.sum(own_strategy * action_values)
                regrets_t = action_values - v
                info_active = game.info_state(state, active)
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

    def train_loop(key, verbose=False):
        rng = jax.random.PRNGKey(config.seed)
        params, opt_states, optim = init_trainer(rng)
        buffers = [
            {"infos": jnp.zeros((0, game.info_state_dim()), dtype=jnp.float32),
             "regrets": jnp.zeros((0, K), dtype=jnp.float32)}
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
                buffers[p]["infos"] = jnp.concatenate([buffers[p]["infos"], new_infos])
                buffers[p]["regrets"] = jnp.concatenate([buffers[p]["regrets"], new_regrets])
                cap = config.buffer_capacity
                if buffers[p]["infos"].shape[0] > cap:
                    buffers[p]["infos"] = buffers[p]["infos"][-cap:]
                    buffers[p]["regrets"] = buffers[p]["regrets"][-cap:]
                buf_infos = buffers[p]["infos"]
                buf_regrets = buffers[p]["regrets"]
                n = buf_infos.shape[0]
                for _ in range(config.train_steps_per_iter):
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
            if verbose and (it % 20 == 0 or it == config.iterations - 1):
                print(f"  iter {it}/{config.iterations}", flush=True)
        return params

    return train_loop, net


def evaluate(game: HansonGame, net: RegretNet, params, n_samples: int = 1500,
              seed: int = 0):
    import jax.random as jr
    rng = jr.PRNGKey(seed)
    correct_total = 0.0
    a_wins_per_omega = [0] * 8
    correct_per_omega = [0] * 8
    for omega_idx in range(8):
        for s in range(n_samples):
            rng, k_init = jr.split(rng)
            state = HansonState(
                omega=jnp.array(omega_idx, dtype=jnp.int32),
                signals=jnp.array([(omega_idx >> 2) & 1,
                                   (omega_idx >> 1) & 1,
                                   omega_idx & 1], dtype=jnp.int32),
                price_a_history=jnp.zeros(game.num_rounds + 1).at[0].set(game.initial_price),
                price_b_history=jnp.zeros(game.num_rounds + 1).at[0].set(game.initial_price),
                action_history=jnp.zeros(game.num_rounds, dtype=jnp.int32),
                cur_step=jnp.array(0, dtype=jnp.int32),
                holdings_shares=jnp.zeros((3, 2), dtype=jnp.float32),
                holdings_cost=jnp.zeros((3, 2), dtype=jnp.float32),
                finished=jnp.array(False),
            )
            # Fix signal mapping for the omega - signals = SIGNAL_TABLE row
            from jax_futarchy.hanson_game import SIGNAL_TABLE
            state = state._replace(signals=SIGNAL_TABLE[omega_idx])
            for _ in range(game.num_rounds):
                active = int(state.cur_step) % 3
                info = game.info_state(state, jnp.array(active))
                regrets = net.apply(params[active], info)
                strategy = regret_matching(regrets)
                rng, sk = jr.split(rng)
                action = int(jr.categorical(sk, jnp.log(strategy + 1e-12)))
                state = game.step(state, jnp.array(action))
            from jax_futarchy.hanson_game import METRIC_A, METRIC_B
            decision_a = bool(state.price_a_history[-1] >= state.price_b_history[-1])
            if decision_a:
                a_wins_per_omega[omega_idx] += 1
                m = int(METRIC_A[omega_idx])
            else:
                m = int(METRIC_B[omega_idx])
            correct_per_omega[omega_idx] += m
        correct_total += correct_per_omega[omega_idx] / n_samples
    by_omega = {}
    for omega_idx in range(8):
        label = chr(ord('a') + omega_idx)
        by_omega[label] = {
            "metric_A": int(METRIC_A[omega_idx]),
            "metric_B": int(METRIC_B[omega_idx]),
            "decision_A_prob": a_wins_per_omega[omega_idx] / n_samples,
            "metric_realised_prob": correct_per_omega[omega_idx] / n_samples,
        }
    return {
        "decision_accuracy": correct_total / 8,
        "by_omega": by_omega,
    }


__all__ = ["HansonDCFRConfig", "make_dcfr", "evaluate"]
