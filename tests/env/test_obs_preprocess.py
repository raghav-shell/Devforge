"""Tests for submission/doom_agent/obs/preprocess.py (module A2, docs/OBSERVATION_SCHEMA.md sections 1-5)."""

from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from doom_agent.obs import FEATURE_SPECS, ObsConfig, ObsPreprocessor, stack_features
from doom_agent.obs import schema

REPO = Path(__file__).resolve().parents[2]
SUBMISSION = REPO / "submission"
FIXTURES = REPO / "tests" / "fixtures"
TWO_ROOMS = REPO / "maps" / "two_rooms.wad"


def fixture_wad(grid: str) -> str:
    """Same naming as tests/fixtures.py (sha1 of the layout), resolved from tests/env/."""
    path = FIXTURES / f"{hashlib.sha1(grid.encode()).hexdigest()[:16]}.wad"
    assert path.exists(), f"missing fixture for layout:\n{grid}"
    return str(path)


# Layouts from tests/test_env.py: an enemy, a door + exit, a key + locked door + exit, a health item + enemy.
ENEMY = "#####\n#P.z#\n#####"
DOOR = "##########\n#P..D...E#\n##########"
KEY_LOCK = "############\n#P.r..R...E#\n############"
HEALTH = "#########\n#P..h..z#\n#########"
SMALL = "#####\n#P..#\n#####"

USE, TL, TR = 7, 4, 5


# ---------------------------------------------------------------------- helpers
def raw_obs(
    dist: float | np.ndarray = 0.5,
    vitals: np.ndarray | None = None,
    n_ent: int = 0,
    ent_d: float = 0.5,
    hit: int = 1,
) -> dict:
    if vitals is None:
        vitals = np.zeros(17, np.float32)
        vitals[0], vitals[2], vitals[3], vitals[4] = 100, 2, 50, 50
    feats = np.zeros((32, 5), np.float32)
    etype = np.zeros(32, np.int64)
    mask = np.zeros(32, np.int8)
    for i in range(n_ent):
        feats[i] = (ent_d, np.sin(0.3 * i), np.cos(0.3 * i), 5.0, -5.0)
        etype[i] = 1 + i % 11
        mask[i] = 1
    return {
        "rgb": np.zeros((120, 160, 3), np.uint8),
        "depth": np.zeros((120, 160), np.uint8),
        "vitals": np.asarray(vitals, np.float32),
        "rays": {
            "distance": np.broadcast_to(np.float32(dist), (64,)).astype(np.float32),
            "hit_type": np.full(64, hit, np.int64),
        },
        "visible_entities": {"features": feats, "entity_type": etype, "mask": mask},
    }


def act(*buttons: int) -> np.ndarray:
    a = np.zeros(9, np.int8)
    a[list(buttons)] = 1
    return a


def check_specs(f: dict[str, np.ndarray]) -> None:
    assert set(f) == set(FEATURE_SPECS)
    for name, spec in FEATURE_SPECS.items():
        x = f[name]
        assert isinstance(x, np.ndarray), name
        assert x.shape == spec.shape, (name, x.shape)
        assert x.dtype == spec.np_dtype, (name, x.dtype)
        if x.dtype != np.bool_:
            assert np.all(np.isfinite(x)), name
        assert np.all(x >= spec.low_array()), name
        assert np.all(x <= spec.high_array()), name
    m = f["ent_mask"]
    assert np.all(f["ent_feat"][~m] == 0.0) and not np.any(np.signbit(f["ent_feat"][~m]))
    assert np.all(f["ent_type"][~m] == 0)
    for i in schema.ACT_AUX_RESERVED_V1:
        assert f["act_aux"][i] == 0.0
    np.testing.assert_array_equal(f["vitals_feat"][schema.VITALS_WEAPON_ONEHOT].sum(), 1.0)


def rollout(wad: str, steps: int, seed: int, episode_ticks: int = 2100):
    """Random-policy rollout on the official env. Returns raw obs, committed actions, done flags, features."""
    from doom_env import make_env

    rng = np.random.default_rng(seed)
    pre = ObsPreprocessor()
    env = make_env(wad, "MAP01", episode_ticks=episode_ticks, render_buffers=False)
    raws, actions, dones, feats = [], [], [], []
    try:
        obs, _ = env.reset(seed=seed)
        pre.reset()
        for _ in range(steps):
            raws.append(copy.deepcopy(obs))
            f = pre.process(obs)
            feats.append(f)
            p = np.array([0.6, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.15, 0.05])
            a = (rng.random(9) < p).astype(np.int8)
            pre.commit_action(a)
            actions.append(a)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
            dones.append(done)
            if done:
                raws.append(copy.deepcopy(obs))
                feats.append(pre.process(obs))  # final_feats from the stale terminal obs
                actions.append(None)
                dones.append(None)
                obs, _ = env.reset(seed=int(rng.integers(0, 10000)))
                pre.reset()
    finally:
        env.close()
    return raws, actions, dones, feats


def replay(raws, actions, dones, pre: ObsPreprocessor) -> list[dict[str, np.ndarray]]:
    pre.reset()
    out = []
    for raw, a, d in zip(raws, actions, dones):
        out.append(pre.process(raw))
        if a is None:  # terminal obs processed; the worker resets
            pre.reset()
        else:
            pre.commit_action(a)
    return out


@pytest.fixture(scope="module")
def two_rooms_run():
    return rollout(str(TWO_ROOMS), 500, seed=11)


# ---------------------------------------------------------------------- real-env spec checks
def test_specs_on_two_rooms(two_rooms_run) -> None:
    _, _, _, feats = two_rooms_run
    assert len(feats) >= 500
    for f in feats:
        check_specs(f)
    hits = np.concatenate([f["ray_hit"] for f in feats])
    assert {0, 1} & set(np.unique(hits).tolist()), "rays never hit anything?"
    assert any(np.any(f["ray_ddist"] != 0) for f in feats), "agent never moved"


@pytest.mark.parametrize(
    "grid, need",
    [(ENEMY, "entity"), (DOOR, "door"), (KEY_LOCK, "locked_door"), (HEALTH, "entity")],
    ids=["enemy", "door", "key_lock", "health"],
)
def test_specs_on_fixture_maps(grid: str, need: str) -> None:
    _, _, _, feats = rollout(fixture_wad(grid), 200, seed=3, episode_ticks=600)
    for f in feats:
        check_specs(f)
    if need == "entity":
        assert any(f["ent_mask"].any() for f in feats), "no visible entity: bounds test would be vacuous"
        seen = [f for f in feats if f["ent_mask"].any()]
        assert all(f["ent_type"][f["ent_mask"]].min() >= 1 for f in seen)
    else:
        code = {"door": 2, "locked_door": 3}[need]
        assert any(np.any(f["ray_hit"] == code) for f in feats), f"no {need} ray hit"


def test_stale_terminal_obs_gives_zero_deltas() -> None:
    from doom_env import make_env

    env = make_env(fixture_wad(SMALL), "MAP01", episode_ticks=40, render_buffers=False)
    pre = ObsPreprocessor()
    try:
        obs, _ = env.reset(seed=0)
        f = pre.process(obs)
        n = 0
        while True:
            a = act(0, TL)  # move + turn so ordinary steps have non-zero deltas
            pre.commit_action(a)
            obs, _, term, trunc, _ = env.step(a)
            f = pre.process(obs)
            n += 1
            if term or trunc:
                break
            assert n < 100
        assert trunc and not term
        check_specs(f)
        assert np.all(f["ray_ddist"] == 0.0)
        assert f["vitals_feat"][25] == 0.0 and f["vitals_feat"][26] == 0.0
        np.testing.assert_array_equal(f["prev_action"], a.astype(np.float32))  # prev_action and t advance
        assert pre.state_dict()["t"] == n + 1
        # The worker resets right after; the next process is a fresh episode.
        pre.reset()
        obs, _ = env.reset(seed=1)
        g = pre.process(obs)
        assert np.all(g["prev_action"] == 0) and g["act_aux"][3] == 0.0
    finally:
        env.close()


# ---------------------------------------------------------------------- determinism / identity
def test_replay_is_bit_identical_across_instances_and_strictness(two_rooms_run) -> None:
    raws, actions, dones, feats = two_rooms_run
    a = replay(raws, actions, dones, ObsPreprocessor())
    b = replay(raws, actions, dones, ObsPreprocessor(strict=False))  # the DoomAgent setting
    for f0, f1, f2 in zip(feats, a, b):
        for k in FEATURE_SPECS:
            np.testing.assert_array_equal(f0[k], f1[k])
            np.testing.assert_array_equal(f0[k], f2[k])
            assert f0[k].tobytes() == f2[k].tobytes()


def test_batching_paths_agree(two_rooms_run) -> None:
    _, _, _, feats = two_rooms_run
    # DoomAgent path: {k: v[None]}; training worker path: stack_features over envs.
    for f in feats[:50]:
        single = stack_features([f])
        for k in FEATURE_SPECS:
            agent = f[k][None]
            assert single[k].dtype == agent.dtype and single[k].shape == agent.shape
            assert single[k].tobytes() == agent.tobytes()
    batch = stack_features(feats[:8])
    for i, f in enumerate(feats[:8]):
        for k in FEATURE_SPECS:
            assert batch[k][i].tobytes() == f[k].tobytes()
    with pytest.raises(ValueError):
        stack_features([])
    bad = dict(feats[0])
    bad["ray_hit"] = bad["ray_hit"].astype(np.int32)
    with pytest.raises(ValueError):
        stack_features([feats[0], bad])


def test_outputs_are_fresh_arrays() -> None:
    pre = ObsPreprocessor()
    raw = raw_obs(0.5, n_ent=3)
    f0 = pre.process(raw)
    snap = {k: v.copy() for k, v in f0.items()}
    for k, v in f0.items():
        assert not np.shares_memory(v, raw["rays"]["distance"]), k
        assert not np.shares_memory(v, raw["visible_entities"]["features"]), k
    pre.commit_action(act(USE))
    raw["rays"]["distance"][:] = 0.25  # mutate raw input after processing
    f1 = pre.process(raw)
    for k in f0:
        np.testing.assert_array_equal(f0[k], snap[k])  # earlier outputs untouched by later calls
        assert not np.shares_memory(f0[k], f1[k])
    f1["prev_action"][:] = 0  # mutating outputs must not change internal state
    assert pre.state_dict()["prev_action"][USE] == 1
    np.testing.assert_allclose(f1["ray_ddist"], np.clip(4 * (0.25 - 0.5), -1, 1))


# ---------------------------------------------------------------------- reset / history semantics
def test_reset_clears_history() -> None:
    pre = ObsPreprocessor()
    v = raw_obs()["vitals"].copy()
    pre.process(raw_obs(0.5, v))
    for i in range(5):
        pre.commit_action(act(0, TL, USE) if i % 2 else act(0, TL))
        v[0] -= 10
        v[3] -= 1
        f = pre.process(raw_obs(0.5 - 0.05 * i, v))
    assert np.any(f["prev_action"]) and f["act_aux"][3] > 0
    pre.reset()
    g = pre.process(raw_obs(0.9, v))
    assert np.all(g["ray_ddist"] == 0)
    assert g["vitals_feat"][25] == 0 and g["vitals_feat"][26] == 0
    assert np.all(g["prev_action"] == 0)
    assert g["act_aux"][0] == 1.0 and g["act_aux"][1] == 0.0 and g["act_aux"][3] == 0.0
    sd = pre.state_dict()
    assert sd["t"] == 1 and sd["steps_since_use_edge"] == 16 and sd["turn_run"] == 0


def test_reset_mid_alternation_allows_process() -> None:
    pre = ObsPreprocessor()
    pre.process(raw_obs())
    pre.reset()  # terminal obs processed, no commit: the vector-env auto-reset path
    pre.process(raw_obs())


def test_vitals_deltas_and_weapon_change() -> None:
    pre = ObsPreprocessor()
    v = raw_obs()["vitals"].copy()
    f = pre.process(raw_obs(vitals=v))
    np.testing.assert_allclose(f["vitals_feat"][0], 1.0)
    assert f["vitals_feat"][2 + 2] == 1.0
    pre.commit_action(act(6))
    v2 = v.copy()
    v2[0], v2[3] = 90, 49  # took 10 damage, fired one shot
    f = pre.process(raw_obs(vitals=v2))
    np.testing.assert_allclose(f["vitals_feat"][25], -0.5)
    np.testing.assert_allclose(f["vitals_feat"][26], -0.2)
    pre.commit_action(act(8))
    v3 = v2.copy()
    v3[2], v3[3] = 3, 8  # switched weapon: ammo jump must not show as a delta
    f = pre.process(raw_obs(vitals=v3))
    assert f["vitals_feat"][26] == 0.0 and f["vitals_feat"][2 + 3] == 1.0
    pre.commit_action(act())
    v4 = v3.copy()
    v4[0] = 500  # huge heal is clipped
    f = pre.process(raw_obs(vitals=v4))
    assert f["vitals_feat"][25] == 1.0 and f["vitals_feat"][0] == 2.0


def test_entity_features_and_padding() -> None:
    pre = ObsPreprocessor()
    f = pre.process(raw_obs(n_ent=4, ent_d=0.5))
    e = f["ent_feat"]
    np.testing.assert_allclose(e[:4, 0], 0.125)
    np.testing.assert_allclose(e[:4, 1], 1 / (1 + 4))
    np.testing.assert_allclose(e[:4, 4], 0.5)
    np.testing.assert_allclose(e[:4, 5], -0.5)
    assert np.all(e[:4, 6] == 1.0)
    assert np.all(e[4:] == 0.0) and np.all(f["ent_type"][4:] == 0)
    assert f["ent_mask"].sum() == 4 and f["ent_mask"].dtype == np.bool_
    pre.commit_action(act())
    g = pre.process(raw_obs(n_ent=2, ent_d=3.0))  # beyond ray range
    assert np.all(g["ent_feat"][:2, 6] == 0.0)
    np.testing.assert_allclose(g["ent_feat"][:2, 0], 0.75)


def test_ray_features() -> None:
    pre = ObsPreprocessor()
    f = pre.process(raw_obs(1.0))
    np.testing.assert_allclose(f["ray_near"], 1 / 17, rtol=1e-6)
    pre.commit_action(act())
    f = pre.process(raw_obs(0.015625))
    np.testing.assert_allclose(f["ray_near"], 0.8, rtol=1e-6)
    assert np.all(f["ray_ddist"] == -1.0)  # clip(4 * (0.0156 - 1)) = -1


# ---------------------------------------------------------------------- action history (ACTION_SPACE section 3)
def test_use_edge_counter() -> None:
    pre = ObsPreprocessor()
    f = pre.process(raw_obs())
    assert f["act_aux"][0] == 1.0  # saturated after reset
    reads = []
    for use in (1, 1, 0, 1):
        pre.commit_action(act(USE) if use else act())
        f = pre.process(raw_obs())
        reads.append(round(float(f["act_aux"][0]) * 16))
        assert f["prev_action"][USE] == use
    assert reads == [0, 1, 2, 0]
    for _ in range(30):  # saturates at 16
        pre.commit_action(act())
        f = pre.process(raw_obs())
    assert f["act_aux"][0] == 1.0


def test_use_held_from_first_action_is_an_edge() -> None:
    pre = ObsPreprocessor()
    pre.process(raw_obs())
    pre.commit_action(act(USE))
    assert pre.process(raw_obs())["act_aux"][0] == 0.0


def test_turn_run() -> None:
    pre = ObsPreprocessor()
    assert pre.process(raw_obs())["act_aux"][1] == 0.0
    reads = []
    for a in (act(TL), act(TL), act(TL), act(TL), act(TR), act()):
        pre.commit_action(a)
        reads.append(float(pre.process(raw_obs())["act_aux"][1]) * 3)
    np.testing.assert_allclose(reads, [1, 2, 3, 3, 3, 0])
    pre.commit_action(act(TL))
    pre.process(raw_obs())
    pre.commit_action(act(TL, TR))  # both pressed cancels
    assert pre.process(raw_obs())["act_aux"][1] == 0.0


def test_time_frac_and_ablation() -> None:
    pre = ObsPreprocessor()
    off = ObsPreprocessor(ObsConfig(time_feature=False))
    for t in range(1200):
        f, g = pre.process(raw_obs()), off.process(raw_obs())
        if t in (0, 1, 525, 1199):
            np.testing.assert_allclose(f["act_aux"][3], min(t / 1050, 1.0), rtol=1e-6)
        assert g["act_aux"][3] == 0.0
        pre.commit_action(act())
        off.commit_action(act())
    scaled = ObsPreprocessor(ObsConfig(time_scale_steps=10.0))
    for _ in range(6):
        f = scaled.process(raw_obs())
        scaled.commit_action(act())
    np.testing.assert_allclose(f["act_aux"][3], 0.5)


# ---------------------------------------------------------------------- assertions and validation
def test_ordering_assertions() -> None:
    pre = ObsPreprocessor()
    with pytest.raises(RuntimeError):
        pre.commit_action(act())
    pre.process(raw_obs())
    with pytest.raises(RuntimeError):
        pre.process(raw_obs())
    pre.commit_action(act())
    with pytest.raises(RuntimeError):
        pre.commit_action(act())


@pytest.mark.parametrize("bad", [np.zeros(8, np.int8), np.full(9, 2, np.int8), np.zeros((6,), np.int64)])
def test_commit_action_rejects_invalid(bad) -> None:
    pre = ObsPreprocessor()
    pre.process(raw_obs())
    with pytest.raises(ValueError):
        pre.commit_action(bad)
    pre.commit_action(act())  # state unchanged by the failed commit


@pytest.mark.parametrize("flag", ["enable_depth_band", "enable_egomap", "enable_icp"])
def test_v2_flags_raise(flag: str) -> None:
    with pytest.raises(NotImplementedError):
        ObsPreprocessor(ObsConfig(**{flag: True}))


def test_extra_keys_strict_and_lenient(capsys) -> None:
    for extra_at in ("top", "rays", "visible_entities"):
        raw = raw_obs()
        if extra_at == "top":
            raw["privileged"] = {"x": 1.0}
        else:
            raw[extra_at]["leak"] = np.zeros(3)
        with pytest.raises(KeyError):
            ObsPreprocessor().process(raw)
        lenient = ObsPreprocessor(strict=False)
        f = lenient.process(raw)
        np.testing.assert_array_equal(f["ray_dist"], ObsPreprocessor().process(raw_obs())["ray_dist"])
        lenient.commit_action(act())
        lenient.process(raw)  # warns only once
    err = capsys.readouterr().err
    assert err.count("WARNING ObsPreprocessor") == 3


@pytest.mark.parametrize("strict", [True, False])
def test_missing_keys_and_bad_shapes_always_raise(strict: bool) -> None:
    raw = raw_obs()
    del raw["vitals"]
    with pytest.raises(KeyError):
        ObsPreprocessor(strict=strict).process(raw)
    raw = raw_obs()
    del raw["visible_entities"]["mask"]
    with pytest.raises(KeyError):
        ObsPreprocessor(strict=strict).process(raw)
    raw = raw_obs()
    raw["rays"]["distance"] = np.zeros(63, np.float32)
    with pytest.raises(ValueError):
        ObsPreprocessor(strict=strict).process(raw)


def test_non_finite_and_bad_categoricals() -> None:
    raw = raw_obs()
    raw["vitals"][0] = np.nan
    with pytest.raises(ValueError):
        ObsPreprocessor().process(raw)
    pre = ObsPreprocessor()
    with pytest.raises(ValueError):
        pre.process(raw)
    pre.process(raw_obs())  # a failed process leaves the instance usable (no state committed)
    f = ObsPreprocessor(strict=False).process(raw)
    check_specs(f)
    raw = raw_obs(hit=7)
    with pytest.raises(ValueError):
        ObsPreprocessor().process(raw)
    check_specs(ObsPreprocessor(strict=False).process(raw))
    raw = raw_obs()
    raw["visible_entities"]["features"][0, 3] = np.inf  # inf velocity in a padded slot still fails loudly
    with pytest.raises(ValueError):
        ObsPreprocessor().process(raw)


def test_dtype_tolerance() -> None:
    # The remote socket path may deliver other numeric dtypes; features must be identical.
    a = raw_obs(0.3, n_ent=2)
    b = copy.deepcopy(a)
    b["rays"]["distance"] = b["rays"]["distance"].astype(np.float64)
    b["rays"]["hit_type"] = b["rays"]["hit_type"].astype(np.int32)
    b["visible_entities"]["mask"] = b["visible_entities"]["mask"].astype(bool)
    b["vitals"] = b["vitals"].tolist()
    fa, fb = ObsPreprocessor().process(a), ObsPreprocessor().process(b)
    for k in FEATURE_SPECS:
        assert fa[k].tobytes() == fb[k].tobytes(), k


# ---------------------------------------------------------------------- import hygiene and latency
def test_import_is_light_with_preprocessor() -> None:
    code = (
        "import sys\n"
        "from doom_agent.obs import ObsPreprocessor, ObsConfig\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "('torch', 'training', 'doom_env', 'vizdoom', 'env', 'gymnasium')]\n"
        "print(','.join(bad))\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SUBMISSION), str(REPO)]))
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd="/", capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_latency_single_thread(two_rooms_run) -> None:
    raws, actions, dones, _ = two_rooms_run
    pre = ObsPreprocessor()
    times = []
    for rep in range(4):
        pre.reset()
        for raw, a in zip(raws, actions):
            t0 = time.perf_counter()
            pre.process(raw)
            if a is None:
                pre.reset()
            else:
                pre.commit_action(a)
            times.append(time.perf_counter() - t0)
    ms = np.array(times[50:]) * 1e3
    p50, p99 = np.percentile(ms, [50, 99])
    print(f"\nMEASURED ObsPreprocessor process+commit: p50 {p50:.4f} ms, p99 {p99:.4f} ms over {ms.size} calls")
    assert p50 < 1.0 and p99 < 5.0  # loose: budget is 40 ms per decision; machine is shared


# ---------------------------------------------------------------------- eval path and exact per-index values
def test_remote_socket_path_gives_identical_features() -> None:
    """Obs sent through the scorer's wire format (doom_env.remote encode/recv_message), with real pixel
    buffers, produce byte-identical features to the in-process obs."""
    import socket
    import threading

    from doom_env import make_env, remote

    rng = np.random.default_rng(5)
    env = make_env(str(TWO_ROOMS), "MAP01", episode_ticks=2100, render_buffers=True)
    local, wire = ObsPreprocessor(), ObsPreprocessor(strict=False)
    a_sock, b_sock = socket.socketpair()
    try:
        obs, _ = env.reset(seed=2)
        assert obs["rgb"].any(), "render_buffers=True should give real pixels"
        for _ in range(120):
            sender = threading.Thread(target=lambda o=obs: a_sock.sendall(remote.encode({"type": "act"}, o)))
            sender.start()
            _, received = remote.recv_message(b_sock, None, 1 << 22)
            sender.join()
            f_local, f_wire = local.process(obs), wire.process(received)
            for k in FEATURE_SPECS:
                assert f_local[k].dtype == f_wire[k].dtype and f_local[k].tobytes() == f_wire[k].tobytes(), k
            a = (rng.random(9) < 0.3).astype(np.int8)
            local.commit_action(a)
            wire.commit_action(a)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    finally:
        a_sock.close()
        b_sock.close()
        env.close()


def test_every_vitals_index_maps_exactly() -> None:
    v = np.array(
        [80, 150, 3, 20, 100, 25, 10, 600, 1, 0, 1, 0, 0, 1, 12.0, -45.0, 3.0], np.float32
    )  # health armor weapon selammo bullets shells rockets cells keys(bc yc rc bs ys rs) vf vr vu
    f = ObsPreprocessor().process(raw_obs(vitals=v))
    exp = np.zeros(27, np.float32)
    exp[0], exp[1] = 0.8, 1.5
    exp[2 + 3] = 1.0
    exp[10], exp[11], exp[12], exp[13], exp[14] = 0.4, 0.5, 0.5, 0.2, 2.0
    exp[15:21] = [1, 0, 1, 0, 0, 1]
    exp[21] = 1.0
    exp[22], exp[23], exp[24] = 1.2, -3.0, 0.3
    np.testing.assert_allclose(f["vitals_feat"], exp, rtol=1e-6, atol=1e-7)
    # Weapon slot is rounded and clipped into 0..7.
    for raw_w, slot in ((6.6, 7), (9.0, 7), (-2.0, 0), (0.4, 0)):
        w = v.copy()
        w[2] = raw_w
        g = ObsPreprocessor().process(raw_obs(vitals=w))["vitals_feat"]
        assert g[2 + slot] == 1.0 and g[2:10].sum() == 1.0, raw_w


def test_entity_sin_cos_and_order_pass_through() -> None:
    raw = raw_obs(n_ent=5, ent_d=0.25)
    f = ObsPreprocessor().process(raw)
    src = raw["visible_entities"]["features"]
    np.testing.assert_array_equal(f["ent_feat"][:5, 2], src[:5, 1])
    np.testing.assert_array_equal(f["ent_feat"][:5, 3], src[:5, 2])
    np.testing.assert_array_equal(f["ent_type"][:5], raw["visible_entities"]["entity_type"][:5])
