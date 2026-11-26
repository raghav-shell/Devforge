"""B2 DoomPolicy tests (docs/MODULE_CONTRACTS.md 3.5)."""

import json
import math

import numpy as np
import pytest
import torch

from doom_agent.models.action_codec import LOGIT_SLICES, NUM_LOGITS
from doom_agent.models.config import PolicyConfig
from doom_agent.models.distributions import FactoredCategorical, select_action
from doom_agent.models.io import features_to_torch
from doom_agent.models.policy import REQUIRED_KEYS, DoomPolicy
from doom_agent.obs.schema import FEATURE_SPECS, dummy_features

VARIANTS = [
    {},
    {"fusion": "film"},
    {"ray_encoder": "mlp"},
    {"entity_encoder": "attnpool"},
    {"action_head": "bernoulli"},
    {"extra_value_heads": ["int"]},
]
VID = ["default", "film", "mlp_rays", "attnpool", "bernoulli", "extra_value"]


def make(seed=0, **kw):
    torch.manual_seed(seed)
    return DoomPolicy(PolicyConfig(**kw))


def feats(batch, seed=0):
    return features_to_torch(dummy_features(batch, np.random.default_rng(seed)))


def perturb_weights(p, seed=123):
    """Fresh init has policy gain 0.01 and FiLM at identity; perturb so the tests are sensitive."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for prm in p.parameters():
            prm.add_(0.1 * torch.randn(prm.shape, generator=g))
    return p


# ------------------------------------------------------------------ shapes / dtypes
@pytest.mark.parametrize("kw", VARIANTS, ids=VID)
@pytest.mark.parametrize("B", [1, 7])
def test_step_shapes(kw, B):
    p = make(**kw)
    x = feats(B)
    out, s = p.step(x, p.initial_state(B), torch.zeros(B, dtype=torch.bool))
    nl = 9 if kw.get("action_head") == "bernoulli" else NUM_LOGITS
    assert out["logits"].shape == (B, nl) and out["logits"].dtype == torch.float32
    assert out["value"].shape == (B,)
    assert out["core"].shape == (B, 256)
    assert s.shape == (B, p.state_size) and s.dtype == torch.float32
    assert set(out["values"]) == set(kw.get("extra_value_heads", []))
    for v in out["values"].values():
        assert v.shape == (B,)
    assert torch.equal(out["core"], s)


@pytest.mark.parametrize("B", [1, 7])
def test_sequence_shapes(B):
    p = make(extra_value_heads=["int"])
    T = 5
    out, s = p.forward_sequence(feats((T, B)), p.initial_state(B), torch.zeros(T, B, dtype=torch.bool))
    assert out["logits"].shape == (T, B, 15)
    assert out["value"].shape == (T, B)
    assert out["values"]["int"].shape == (T, B)
    assert out["core"].shape == (T, B, 256)
    assert s.shape == (B, 256)
    assert torch.equal(out["core"][-1], s)


def test_initial_state_and_inputs_from_spec():
    p = make()
    s = p.initial_state(3)
    assert s.shape == (3, 256) and s.dtype == torch.float32 and not s.any()
    assert set(REQUIRED_KEYS) == set(FEATURE_SPECS)
    x = feats(2)
    for k, spec in FEATURE_SPECS.items():
        assert x[k].shape == (2,) + spec.shape
    x.pop("act_aux")
    with pytest.raises(KeyError):
        p.step(x, p.initial_state(2), torch.zeros(2, dtype=torch.bool))


def test_features_to_torch_dtypes():
    f = dummy_features(4)
    f["ray_dist"] = f["ray_dist"].astype(np.float64)
    x = features_to_torch(f, "cpu")
    for k, spec in FEATURE_SPECS.items():
        assert x[k].dtype == torch.from_numpy(np.zeros(0, spec.dtype)).dtype, k
        assert tuple(x[k].shape) == (4,) + spec.shape


def test_hidden_state_is_not_module_state():
    p = make()
    x = feats(2)
    s = torch.randn(2, 256)
    es = torch.zeros(2, dtype=torch.bool)
    o1, _ = p.step(x, s, es)
    o2, _ = p.step(x, s, es)
    assert torch.equal(o1["logits"], o2["logits"])
    o_train, _ = p.train().step(x, s, es)
    o_eval, _ = p.eval().step(x, s, es)
    assert torch.equal(o_train["logits"], o_eval["logits"])


# ------------------------------------------------------------------ entities
@pytest.mark.parametrize("kw", VARIANTS[:4], ids=VID[:4])
def test_entity_permutation_invariance(kw):
    p = perturb_weights(make(**kw))
    x = feats(6, seed=1)
    s = torch.randn(6, 256)
    es = torch.zeros(6, dtype=torch.bool)
    ref, _ = p.step(x, s, es)
    for seed in range(3):
        perm = torch.randperm(FEATURE_SPECS["ent_mask"].shape[0], generator=torch.Generator().manual_seed(seed))
        y = dict(x)
        for k in ("ent_feat", "ent_type", "ent_mask"):
            y[k] = x[k][:, perm]
        out, _ = p.step(y, s, es)
        assert torch.allclose(out["logits"], ref["logits"], atol=1e-5)
        assert torch.allclose(out["value"], ref["value"], atol=1e-5)
        z1, z2 = p.ent_enc(x), p.ent_enc(y)
        assert torch.allclose(z1, z2, atol=1e-5)


@pytest.mark.parametrize("kw", VARIANTS[:4], ids=VID[:4])
def test_garbage_in_masked_slots_ignored(kw):
    p = perturb_weights(make(**kw))
    x = feats(5, seed=2)
    s = torch.randn(5, 256)
    es = torch.zeros(5, dtype=torch.bool)
    ref, _ = p.step(x, s, es)
    y = dict(x)
    m = x["ent_mask"].unsqueeze(-1)
    garbage = torch.full_like(x["ent_feat"], float("nan"))
    garbage[..., 0] = float("inf")
    garbage[..., 1] = 1e30
    y["ent_feat"] = torch.where(m, x["ent_feat"], garbage)
    y["ent_type"] = torch.where(x["ent_mask"], x["ent_type"], torch.full_like(x["ent_type"], 7))
    out, _ = p.step(y, s, es)
    assert torch.equal(out["logits"], ref["logits"])
    assert torch.equal(out["value"], ref["value"])


@pytest.mark.parametrize("kw", VARIANTS[:4], ids=VID[:4])
def test_zero_entities_finite_outputs_and_grads(kw):
    p = perturb_weights(make(**kw))
    x = feats(4, seed=3)
    x["ent_mask"] = x["ent_mask"].clone()
    x["ent_mask"][:2] = False  # rows 0,1 empty; rows 2,3 mixed
    x["ent_feat"] = torch.where(x["ent_mask"].unsqueeze(-1), x["ent_feat"], torch.full_like(x["ent_feat"], float("nan")))
    out, s = p.step(x, torch.zeros(4, 256), torch.zeros(4, dtype=torch.bool))
    assert torch.isfinite(out["logits"]).all() and torch.isfinite(out["value"]).all()
    loss = out["logits"].square().sum() + out["value"].square().sum()
    loss.backward()
    for name, prm in p.named_parameters():
        if prm.grad is not None:
            assert torch.isfinite(prm.grad).all(), name
    # fully empty batch too
    x2 = feats(3, seed=4)
    x2["ent_mask"] = torch.zeros_like(x2["ent_mask"])
    z = p.ent_enc(x2)
    assert torch.isfinite(z).all()


# ------------------------------------------------------------------ rays
def test_ray_conv_circular_shift_equivariance():
    p = perturb_weights(make())
    enc = p.ray_enc
    x = feats(3, seed=5)
    stride = enc.total_stride
    assert stride == 4
    base = enc.features(x)
    for k in (1, 2, 5):
        y = dict(x)
        for key in ("ray_dist", "ray_near", "ray_ddist", "ray_hit"):
            y[key] = torch.roll(x[key], shifts=stride * k, dims=-1)
        shifted = enc.features(y)
        assert torch.allclose(shifted, torch.roll(base, shifts=k, dims=-1), atol=1e-5)


# ------------------------------------------------------------------ recurrence
def _random_starts(T, B, seed):
    g = torch.Generator().manual_seed(seed)
    es = torch.rand(T, B, generator=g) < 0.15
    es[0, 0] = True
    es[T // 2, :] = True  # at least one all-env mid-chunk reset
    es[T // 3, 1] = True
    return es


@pytest.mark.parametrize("kw", VARIANTS, ids=VID)
def test_step_vs_sequence_equivalence(kw):
    p = perturb_weights(make(**kw))
    T, B = 32, 4
    x = feats((T, B), seed=6)
    es = _random_starts(T, B, seed=0)
    s0 = torch.randn(B, 256)
    seq, s_seq = p.forward_sequence(x, s0, es)
    s = s0
    logits, values, extra = [], [], []
    for t in range(T):
        out, s = p.step({k: v[t] for k, v in x.items()}, s, es[t])
        logits.append(out["logits"])
        values.append(out["value"])
        extra.append(out["values"])
    assert (torch.stack(logits) - seq["logits"]).abs().max().item() < 1e-5
    assert (torch.stack(values) - seq["value"]).abs().max().item() < 1e-5
    assert (s - s_seq).abs().max().item() < 1e-5
    for name in seq["values"]:
        assert (torch.stack([e[name] for e in extra]) - seq["values"][name]).abs().max().item() < 1e-5


def test_sequence_chunking_equivalence():
    """Two chunks of 16 with state carried == one chunk of 32 (what PPO's chunked BPTT relies on)."""
    p = perturb_weights(make())
    T, B = 32, 3
    x = feats((T, B), seed=7)
    es = _random_starts(T, B, seed=1)
    s0 = torch.randn(B, 256)
    full, sf = p.forward_sequence(x, s0, es)
    a, sa = p.forward_sequence({k: v[:16] for k, v in x.items()}, s0, es[:16])
    b, sb = p.forward_sequence({k: v[16:] for k, v in x.items()}, sa, es[16:])
    assert torch.allclose(torch.cat([a["logits"], b["logits"]]), full["logits"], atol=1e-5)
    assert torch.allclose(sb, sf, atol=1e-5)


def test_reset_isolation():
    p = perturb_weights(make())
    T, B, t0 = 20, 3, 8
    x1 = feats((T, B), seed=8)
    x2 = feats((T, B), seed=9)
    x2 = {k: torch.cat([x2[k][:t0], x1[k][t0:]]) for k in x1}  # different prefix, same suffix
    es = torch.zeros(T, B, dtype=torch.bool)
    es[t0] = True
    o1, _ = p.forward_sequence(x1, torch.randn(B, 256), es)
    o2, _ = p.forward_sequence(x2, torch.randn(B, 256) * 5, es)
    assert not torch.allclose(o1["logits"][:t0], o2["logits"][:t0])
    assert torch.equal(o1["logits"][t0:], o2["logits"][t0:])
    assert torch.equal(o1["value"][t0:], o2["value"][t0:])
    # step-level: start=True with a garbage state == zero state
    x = feats(B, seed=10)
    a, _ = p.step(x, torch.full((B, 256), float("nan")), torch.ones(B, dtype=torch.bool))
    b, _ = p.step(x, torch.zeros(B, 256), torch.zeros(B, dtype=torch.bool))
    assert torch.equal(a["logits"], b["logits"])
    # per-env reset: only env 1 is reset
    s = torch.randn(B, 256)
    es1 = torch.tensor([False, True, False])
    c, _ = p.step(x, s, es1)
    s_masked = s.clone()
    s_masked[1] = 0
    d, _ = p.step(x, s_masked, torch.zeros(B, dtype=torch.bool))
    assert torch.equal(c["logits"], d["logits"])


def test_no_future_leakage():
    p = perturb_weights(make())
    T, B, t1 = 16, 2, 9
    x = feats((T, B), seed=11)
    y = {k: v.clone() for k, v in x.items()}
    y["ray_dist"][t1:] = torch.rand_like(y["ray_dist"][t1:])
    y["vitals_feat"][t1:] = 0.0
    es = torch.zeros(T, B, dtype=torch.bool)
    s0 = torch.randn(B, 256)
    a, _ = p.forward_sequence(x, s0, es)
    b, _ = p.forward_sequence(y, s0, es)
    assert torch.equal(a["logits"][:t1], b["logits"][:t1])
    assert not torch.allclose(a["logits"][t1:], b["logits"][t1:])


# ------------------------------------------------------------------ init / heads
def test_head_bias_init_factored():
    p = make()
    with torch.no_grad():
        logits = p.actor(torch.zeros(1, 256))[0]
    d = FactoredCategorical(logits)
    probs = d.probs()
    p_use, p_nw = probs[4][1].item(), probs[5][1].item()
    expect = 1 / (1 + math.exp(2.0))
    assert abs(p_use - expect) < 1e-6 and abs(p_nw - expect) < 1e-6
    assert abs(expect - 0.1192) < 1e-4
    for k in range(4):  # move/strafe/turn/attack uniform
        assert torch.allclose(probs[k], torch.full_like(probs[k], 1 / probs[k].numel()))
    assert p.actor.bias[LOGIT_SLICES[4].start].item() == 0.0


def test_head_bias_init_bernoulli_and_ablation():
    p = make(action_head="bernoulli")
    with torch.no_grad():
        pr = torch.sigmoid(p.actor(torch.zeros(1, 256))[0])
    assert abs(pr[7].item() - 0.1192) < 1e-4 and abs(pr[8].item() - 0.1192) < 1e-4
    assert torch.allclose(pr[:7], torch.full((7,), 0.5))
    p0 = make(head_bias_init={"use": -2.0, "next_weapon": 0.0})  # D14 cheap ablation
    assert p0.actor.bias[14].item() == 0.0 and p0.actor.bias[12].item() == -2.0
    with pytest.raises(ValueError):
        make(head_bias_init={"move": -1.0})
    with pytest.raises(ValueError):
        make(head_bias_init={"jump": -1.0})


def test_init_policy_near_uniform():
    p = make()
    out, _ = p.step(feats(8), p.initial_state(8), torch.ones(8, dtype=torch.bool))
    ent = FactoredCategorical(out["logits"]).entropy_per_head()
    assert (ent[:, :4].sum(-1) > (3 * math.log(3) + math.log(2)) - 0.01).all()


def test_reserved_configs_raise():
    for kw in ({"core": "lstm"}, {"entity_encoder": "transformer"}, {"use_egomap": True}, {"use_depth_band": True}):
        with pytest.raises(NotImplementedError):
            make(**kw)
    with pytest.raises(ValueError):
        make(fusion="xattn")


# ------------------------------------------------------------------ modes
def test_deterministic_vs_stochastic_modes():
    p = perturb_weights(make())
    p.eval()
    x = feats(1, seed=12)
    with torch.inference_mode():
        out, _ = p.step(x, p.initial_state(1), torch.ones(1, dtype=torch.bool))
    lg = out["logits"][0].numpy()
    greedy = [select_action(lg, "greedy", 1.0, None) for _ in range(20)]
    assert all(np.array_equal(g, greedy[0]) for g in greedy)
    r1, r2 = np.random.default_rng([0, 1]), np.random.default_rng([0, 1])
    s1 = np.stack([select_action(lg, "sample", 1.0, r1) for _ in range(200)])
    s2 = np.stack([select_action(lg, "sample", 1.0, r2) for _ in range(200)])
    np.testing.assert_array_equal(s1, s2)
    assert len({tuple(r) for r in s1}) > 1  # sampling actually varies
    # torch-side helpers agree on the mode
    d = p.distribution(out["logits"])
    from doom_agent.models.action_codec import factors_to_buttons

    np.testing.assert_array_equal(factors_to_buttons(d.mode()[0].numpy()), greedy[0])
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    assert torch.equal(d.sample(g1), d.sample(g2))


# ------------------------------------------------------------------ serialisation
@pytest.mark.parametrize("kw", VARIANTS, ids=VID)
def test_state_dict_strict_roundtrip(kw, tmp_path):
    p = perturb_weights(make(**kw))
    cfg_json = json.dumps(p.cfg.to_dict())
    torch.save(p.state_dict(), tmp_path / "policy.pt")
    cfg2 = PolicyConfig.from_dict(json.loads(cfg_json))
    assert cfg2 == p.cfg
    torch.manual_seed(999)
    q = DoomPolicy(cfg2)
    sd = torch.load(tmp_path / "policy.pt", weights_only=True)
    q.load_state_dict(sd, strict=True)
    x = feats(3, seed=13)
    s = torch.randn(3, 256)
    es = torch.zeros(3, dtype=torch.bool)
    a, _ = p.step(x, s, es)
    b, _ = q.step(x, s, es)
    assert torch.equal(a["logits"], b["logits"]) and torch.equal(a["value"], b["value"])
    assert not any(("aux" in k) or ("priv" in k) for k in sd)


def test_config_from_dict_rejects_unknown():
    d = PolicyConfig().to_dict()
    d["bogus"] = 1
    with pytest.raises(ValueError):
        PolicyConfig.from_dict(d)
    d = PolicyConfig().to_dict()
    assert d == {
        "hidden_size": 256, "core": "gru", "ray_encoder": "circconv", "entity_encoder": "deepsets",
        "fusion": "concat", "action_head": "factored", "head_bias_init": {"use": -2.0, "next_weapon": -2.0},
        "extra_value_heads": [], "use_egomap": False, "use_depth_band": False,
        "obs_schema_version": "1.0.0", "action_space_version": "1.0.0",
    }


def test_param_count_v1():
    n = make().num_params()
    assert 600_000 < n < 650_000, n  # contract measured ~625k


def test_circular_conv_matches_native_conv1d():
    import torch.nn.functional as F

    from doom_agent.models.encoders import CircularConv1d

    g = torch.Generator().manual_seed(0)
    for cin, cout, k, s in ((7, 32, 5, 1), (32, 32, 5, 2), (32, 64, 3, 2)):
        conv = CircularConv1d(cin, cout, k, stride=s)
        x = torch.randn(5, cin, 64 // (1 if s == 1 else 2), generator=g, requires_grad=True)
        ref = F.conv1d(F.pad(x, (k // 2, k // 2), mode="circular"), conv.weight, conv.bias, stride=s)
        out = conv(x)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5)
        gx_ref = torch.autograd.grad(ref.square().sum(), x)[0]
        gx = torch.autograd.grad(out.square().sum(), x)[0]
        assert torch.allclose(gx, gx_ref, atol=1e-4)
        assert torch.allclose(conv(x[:1].detach()), out[:1], atol=1e-5)  # batch-1 acting path
        native = torch.nn.Conv1d(cin, cout, k, stride=s, padding=k // 2, padding_mode="circular")
        assert set(native.state_dict()) == set(conv.state_dict())
