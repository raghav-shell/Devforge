"""GAE against hand-computed examples (MODULE_CONTRACTS 4.2, RL_RESEARCH 3.3)."""

import numpy as np
import pytest

from training.ppo.gae import compute_gae

G, L = 0.9, 0.8  # gamma, lambda: gamma*lambda = 0.72


def col(*xs, dtype=np.float32):
    return np.array(xs, dtype=dtype)[:, None]


def run(r, v, term, trunc, tv, last):
    return compute_gae(col(*r), col(*v), col(*term, dtype=bool), col(*trunc, dtype=bool), col(*tv),
                       np.array([last], np.float32), G, L)


def test_end_of_rollout_bootstrap():
    # no done: A1 = 1 + .9*2 - .5 = 2.3 ; A0 = (1 + .9*.5 - .5) + .72*2.3 = 0.95 + 1.656 = 2.606
    adv, ret = run([1, 1], [.5, .5], [0, 0], [0, 0], [0, 0], 2.0)
    np.testing.assert_allclose(adv[:, 0], [2.606, 2.3], atol=1e-6)
    np.testing.assert_allclose(ret[:, 0], [3.106, 2.8], atol=1e-6)


def test_terminated_no_bootstrap_and_cut():
    # terminated at t=1: A2 = 1 + .9*2 - .5 = 2.3 ; A1 = 1 - .5 = .5 ; A0 = .95 + .72*.5 = 1.31
    adv, _ = run([1, 1, 1], [.5, .5, .5], [0, 1, 0], [0, 0, 0], [0, 0, 0], 2.0)
    np.testing.assert_allclose(adv[:, 0], [1.31, 0.5, 2.3], atol=1e-6)


def test_terminated_ignores_trunc_value():
    adv, _ = run([1, 1, 1], [.5, .5, .5], [0, 1, 0], [0, 0, 0], [0, 99.0, 0], 2.0)
    np.testing.assert_allclose(adv[:, 0], [1.31, 0.5, 2.3], atol=1e-6)


def test_truncated_bootstraps_from_trunc_value_not_next_obs():
    # truncated at t=1 with V(final_obs)=3: A1 = 1 + .9*3 - .5 = 3.2 ; A0 = .95 + .72*3.2 = 3.254
    # V_2 (=.5, first obs of the NEXT episode) must not enter A1.
    adv, _ = run([1, 1, 1], [.5, .5, .5], [0, 0, 0], [0, 1, 0], [0, 3.0, 0], 2.0)
    np.testing.assert_allclose(adv[:, 0], [3.254, 3.2, 2.3], atol=1e-6)
    adv2, _ = run([1, 1, 1], [.5, .5, 7.0], [0, 0, 0], [0, 1, 0], [0, 3.0, 0], 2.0)
    np.testing.assert_allclose(adv2[:2, 0], [3.254, 3.2], atol=1e-6)


def test_both_flags_terminated_wins():
    adv, _ = run([1, 1], [.5, .5], [1, 0], [1, 0], [5.0, 0], 2.0)
    assert adv[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_crash_truncation_zero_trunc_value():
    # crash -> truncated with trunc_value 0: A1 = 1 - .5 = .5
    adv, _ = run([1, 1, 1], [.5, .5, .5], [0, 0, 0], [0, 1, 0], [0, 0.0, 0], 2.0)
    np.testing.assert_allclose(adv[:, 0], [1.31, 0.5, 2.3], atol=1e-6)


def test_mid_rollout_reset_isolates_episodes():
    # Two episodes in one rollout (term at t=1). Changing the second episode's rewards/values
    # must not change the first episode's advantages.
    a1, _ = run([1, 1, 1, 1], [.5, .5, .5, .5], [0, 1, 0, 0], [0] * 4, [0] * 4, 2.0)
    a2, _ = run([1, 1, -5, 8], [.5, .5, 3, -2], [0, 1, 0, 0], [0] * 4, [0] * 4, -9.0)
    np.testing.assert_allclose(a1[:2], a2[:2], atol=1e-7)


def test_constant_reward_truncation_analytic():
    # lambda=1, V=0, r=1, truncation at t=k with trunc_value c:
    # A_t = sum_{j=t..k} g^(j-t) + g^(k-t+1) * c  for t <= k
    g, k, c, T = 0.97, 5, 4.0, 9
    r = np.ones((T, 1), np.float32)
    v = np.zeros((T, 1), np.float32)
    trunc = np.zeros((T, 1), bool)
    trunc[k] = True
    tv = np.zeros((T, 1), np.float32)
    tv[k] = c
    adv, _ = compute_gae(r, v, np.zeros_like(trunc), trunc, tv, np.zeros(1, np.float32), g, 1.0)
    for t in range(k + 1):
        expect = sum(g ** (j - t) for j in range(t, k + 1)) + g ** (k - t + 1) * c
        assert adv[t, 0] == pytest.approx(expect, rel=1e-5)


def test_matches_bruteforce_random():
    rng = np.random.default_rng(3)
    T, N, g, lam = 40, 5, 0.99, 0.9
    r = rng.normal(size=(T, N)).astype(np.float32)
    v = rng.normal(size=(T, N)).astype(np.float32)
    term = rng.random((T, N)) < 0.08
    trunc = (rng.random((T, N)) < 0.08) & ~term
    tv = np.where(trunc, rng.normal(size=(T, N)), 0).astype(np.float32)
    last = rng.normal(size=N).astype(np.float32)
    adv, _ = compute_gae(r, v, term, trunc, tv, last, g, lam)
    # brute force: A_t = sum_l (g*lam)^l delta_{t+l} within the episode
    vn = np.concatenate([v[1:], last[None]], 0).astype(np.float64)
    for n in range(N):
        for t in range(T):
            tot, w = 0.0, 1.0
            for j in range(t, T):
                if term[j, n]:
                    nv = 0.0
                elif trunc[j, n]:
                    nv = tv[j, n]
                else:
                    nv = vn[j, n]
                tot += w * (r[j, n] + g * nv - v[j, n])
                if term[j, n] or trunc[j, n]:
                    break
                w *= g * lam
            assert adv[t, n] == pytest.approx(tot, rel=1e-4, abs=1e-5)


def test_shape_validation():
    with pytest.raises(ValueError):
        compute_gae(np.zeros((3, 2)), np.zeros((3, 1)), np.zeros((3, 2), bool), np.zeros((3, 2), bool),
                    np.zeros((3, 2)), np.zeros(2), 0.9, 0.9)
