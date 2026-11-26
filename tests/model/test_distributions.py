"""B1 distribution + select_action tests (docs/ACTION_SPACE.md 2.2, 4, 6)."""

import math

import numpy as np
import pytest
import torch

from doom_agent.models.action_codec import HEAD_SIZES, LOGIT_SLICES, buttons_to_factors, factors_to_buttons
from doom_agent.models.distributions import (
    BernoulliHeads,
    FactoredCategorical,
    actions_to_buttons,
    select_action,
)

MAX_ENT = 3 * math.log(3) + 3 * math.log(2)


def test_entropy_at_zero_logits():
    d = FactoredCategorical(torch.zeros(4, 15))
    assert torch.allclose(d.entropy(), torch.full((4,), MAX_ENT), atol=1e-5)
    eph = d.entropy_per_head()
    assert eph.shape == (4, 6)
    assert torch.allclose(eph.sum(-1), d.entropy())
    b = BernoulliHeads(torch.zeros(2, 9))
    assert torch.allclose(b.entropy(), torch.full((2,), 9 * math.log(2)), atol=1e-5)


def test_log_prob_matches_manual_and_normalises():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 15, generator=g)
    d = FactoredCategorical(logits)
    idx = d.sample(generator=g)
    assert idx.shape == (3, 6) and idx.dtype == torch.int64
    manual = sum(torch.log_softmax(logits[:, s], -1).gather(-1, idx[:, k : k + 1]).squeeze(-1)
                 for k, s in enumerate(LOGIT_SLICES))
    assert torch.allclose(d.log_prob(idx), manual, atol=1e-6)
    assert torch.isfinite(d.log_prob(idx)).all()
    # joint probabilities over all 216 actions sum to 1 for each row
    from doom_agent.models.action_codec import all_factor_tuples

    allidx = torch.from_numpy(all_factor_tuples())
    lp = FactoredCategorical(logits[:1].expand(216, 15)).log_prob(allidx)
    assert abs(lp.exp().sum().item() - 1.0) < 1e-5
    # entropy == -E[log p] exactly over the joint
    assert abs(-(lp.exp() * lp).sum().item() - FactoredCategorical(logits[:1]).entropy().item()) < 1e-5


def test_sample_frequencies_match_probs():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(15, generator=g) * 2
    d = FactoredCategorical(logits.expand(40000, 15))
    s = d.sample(generator=g)
    for k, lp in enumerate(d.head_log_probs()):
        freq = torch.bincount(s[:, k], minlength=HEAD_SIZES[k]).float() / s.shape[0]
        assert torch.allclose(freq, lp[0].exp(), atol=0.01), k


def test_mode_and_ties():
    d = FactoredCategorical(torch.zeros(15))
    assert d.mode().tolist() == [0] * 6  # ties -> lowest index ("none")
    logits = torch.zeros(15)
    logits[1] = 5.0  # move fwd
    logits[13 + 1] = 5.0  # next_weapon on
    assert FactoredCategorical(logits).mode().tolist() == [1, 0, 0, 0, 0, 1]
    assert BernoulliHeads(torch.zeros(9)).mode().sum().item() == 0


def test_temperature():
    logits = torch.randn(5, 15)
    hot = FactoredCategorical(logits, temperature=100.0).entropy()
    cold = FactoredCategorical(logits, temperature=0.01).entropy()
    base = FactoredCategorical(logits).entropy()
    assert (hot > base).all() and (cold < base).all()
    with pytest.raises(ValueError):
        FactoredCategorical(logits, temperature=0.0)


def test_bernoulli_logprob_sample():
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(30000, 9, generator=g)
    d = BernoulliHeads(logits)
    a = d.sample(generator=g)
    assert a.shape == (30000, 9) and set(a.unique().tolist()) <= {0, 1}
    manual = -torch.nn.functional.binary_cross_entropy_with_logits(logits, a.float(), reduction="none").sum(-1)
    assert torch.allclose(d.log_prob(a), manual, atol=1e-5)
    np.testing.assert_array_equal(actions_to_buttons(a[:3].numpy(), "bernoulli"), a[:3].numpy().astype(np.int8))


def test_wrong_logit_width_raises():
    with pytest.raises(ValueError):
        FactoredCategorical(torch.zeros(9))
    with pytest.raises(ValueError):
        BernoulliHeads(torch.zeros(15))


# ---------------------------------------------------------------- numpy select_action
def test_select_action_greedy_deterministic_and_matches_torch_mode():
    rng = np.random.default_rng(0)
    for _ in range(50):
        logits = rng.normal(size=15).astype(np.float32)
        a1 = select_action(logits, "greedy", 1.0, None)
        a2 = select_action(logits, "greedy", 1.0, None)
        np.testing.assert_array_equal(a1, a2)
        assert a1.dtype == np.int8 and a1.shape == (9,)
        mode = FactoredCategorical(torch.from_numpy(logits)).mode().numpy()
        np.testing.assert_array_equal(a1, factors_to_buttons(mode))


def test_select_action_sample_reproducible_and_contract_draw_order():
    logits = np.random.default_rng(3).normal(size=15).astype(np.float32)
    s1 = [select_action(logits, "sample", 1.0, r) for r in [np.random.default_rng([7, 1])] for _ in range(1)]
    r1, r2 = np.random.default_rng([7, 1]), np.random.default_rng([7, 1])
    seq1 = np.stack([select_action(logits, "sample", 0.7, r1) for _ in range(100)])
    seq2 = np.stack([select_action(logits, "sample", 0.7, r2) for _ in range(100)])
    np.testing.assert_array_equal(seq1, seq2)
    assert s1[0].dtype == np.int8
    # reference implementation of ACTION_SPACE.md section 4, written out independently
    r = np.random.default_rng([7, 1])
    for row in seq1:
        idx = []
        for k, s in enumerate(LOGIT_SLICES):
            z = logits[s].astype(np.float64) / 0.7
            p = np.exp(z - z.max())
            p /= p.sum()
            idx.append(min(int(np.searchsorted(np.cumsum(p), r.random(), side="right")), HEAD_SIZES[k] - 1))
        np.testing.assert_array_equal(row, factors_to_buttons(np.array(idx)))


def test_select_action_sample_frequencies_match_torch_probs():
    logits = np.random.default_rng(4).normal(size=15).astype(np.float32) * 2
    rng = np.random.default_rng(0)
    acts = np.stack([buttons_to_factors(select_action(logits, "sample", 1.0, rng)) for _ in range(20000)])
    probs = FactoredCategorical(torch.from_numpy(logits)).probs()
    for k in range(6):
        freq = np.bincount(acts[:, k], minlength=HEAD_SIZES[k]) / len(acts)
        np.testing.assert_allclose(freq, probs[k].numpy(), atol=0.015)


def test_select_action_bernoulli():
    logits = np.array([3, -3, 0, 0.1, -0.1, 5, -5, 1, -1], dtype=np.float32)
    np.testing.assert_array_equal(select_action(logits, "greedy", 1.0, None, "bernoulli"),
                                  (logits > 0).astype(np.int8))
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    a = select_action(logits, "sample", 1.0, r1, "bernoulli")
    u = r2.random(9)
    np.testing.assert_array_equal(a, (u < 1 / (1 + np.exp(-logits.astype(np.float64)))).astype(np.int8))


def test_select_action_validation():
    with pytest.raises(ValueError):
        select_action(np.zeros(15), "argmax", 1.0, None)
    with pytest.raises(ValueError):
        select_action(np.zeros(15), "sample", 1.0, None)
    with pytest.raises(ValueError):
        select_action(np.zeros(9), "greedy", 1.0, None)  # factored needs 15
    with pytest.raises(ValueError):
        select_action(np.zeros(15), "greedy", 1.0, None, "bernoulli")
