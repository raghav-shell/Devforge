"""Tests for training/rewards/base.py (module C1, docs/MODULE_CONTRACTS.md section 6; D10 first occurrence)."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from training.rewards import base
from training.rewards.base import (
    REWARD_TERMS,
    BaseRewardTerm,
    FirstOccurrence,
    RewardComposer,
    RewardTerm,
    StepContext,
    load_term_modules,
    new_set_bits,
    register_term,
)

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------- fakes (never left in the global registry)
class ConstTerm(BaseRewardTerm):
    name = "fake_const"

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.c = float(value)
        self.resets = 0

    def on_reset(self, priv0, obs0, inventory, spec) -> None:
        self.resets += 1

    def value(self, ctx: StepContext) -> float:
        return self.c


class KeyFirstTerm(BaseRewardTerm):
    """Pays 1 per key bit the first time it is set this episode (D10)."""

    name = "fake_key_first"

    def __init__(self) -> None:
        super().__init__()
        self.seen = self.ledger()

    def value(self, ctx: StepContext) -> float:
        return float(len(new_set_bits(ctx.priv["keys"], self.seen)))


class DoorFirstTerm(BaseRewardTerm):
    """Pays 1 the first time each door sector reads open (opening >= 56) this episode."""

    name = "fake_door_first"

    def __init__(self, threshold: float = 56.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.opened = self.ledger()

    def value(self, ctx: StepContext) -> float:
        opening = np.asarray(ctx.priv["sector_ceiling"]) - np.asarray(ctx.priv["sector_floor"])
        open_now = [int(i) for i in np.flatnonzero(opening >= self.threshold)]
        return float(len(self.opened.update(open_now)))


class StepTerm:
    """Plain Protocol implementer, no base class: returns ctx.step."""

    name = "fake_step"

    def __init__(self) -> None:
        pass

    def reset(self, priv0, obs0, inventory, spec) -> None:
        pass

    def __call__(self, ctx: StepContext) -> float:
        return np.float32(ctx.step)  # numpy scalar must come back as a Python float


class NanTerm(BaseRewardTerm):
    name = "fake_nan"

    def value(self, ctx: StepContext) -> float:
        return math.nan


FAKES = {c.name: c for c in (ConstTerm, KeyFirstTerm, DoorFirstTerm, StepTerm, NanTerm)}


@pytest.fixture
def reg(monkeypatch: pytest.MonkeyPatch) -> dict:
    for name, cls in FAKES.items():
        monkeypatch.setitem(REWARD_TERMS, name, cls)
    return REWARD_TERMS


def priv(keys=(0, 0, 0, 0, 0, 0), floor=(0.0, 0.0, 0.0), ceil=(128.0, 0.0, 0.0), **kw) -> dict:
    d = {
        "tic": 0, "x": 0.0, "y": 0.0, "kills": 0, "damage_taken": 0.0, "dead": False, "exit_reached": False,
        "keys": np.asarray(keys, np.float32),
        "sector_floor": np.asarray(floor, np.float64),
        "sector_ceiling": np.asarray(ceil, np.float64),
    }
    d.update(kw)
    return d


def ctx(p_prev: dict, p: dict, step: int = 0, terminated=False, truncated=False) -> StepContext:
    return StepContext(
        priv_prev=p_prev, priv=p, obs={}, action_mb=np.zeros(9, np.int8), terminated=terminated,
        truncated=truncated, env_reward=0.0, info={"tic": 4 * step, "exit_reached": False}, step=step,
    )


def start(comp: RewardComposer, p0: dict) -> None:
    comp.reset(p0, {}, {"cells_total": 10}, None)


# ---------------------------------------------------------------------- composition and weights
def test_total_is_weighted_sum_and_per_term_is_unweighted(reg) -> None:
    comp = RewardComposer([
        {"name": "fake_const", "weight": 0.5, "params": {"value": 2.0}},
        {"name": "fake_step", "weight": -0.25},
    ])
    p = priv()
    start(comp, p)
    for step in range(5):
        total, terms = comp(ctx(p, p, step))
        assert terms == {"fake_const": 2.0, "fake_step": float(step)}
        assert all(type(v) is float for v in terms.values())
        assert type(total) is float
        assert total == pytest.approx(0.5 * 2.0 - 0.25 * step)
    assert comp.names == ["fake_const", "fake_step"]
    assert comp.weights == {"fake_const": 0.5, "fake_step": -0.25}


def test_zero_weight_term_still_logged(reg) -> None:
    comp = RewardComposer([{"name": "fake_const", "weight": 0.0, "params": {"value": 3.0}}])
    start(comp, priv())
    total, terms = comp(ctx(priv(), priv()))
    assert total == 0.0 and terms == {"fake_const": 3.0}


def test_empty_composer(reg) -> None:
    comp = RewardComposer([])
    start(comp, priv())
    assert comp(ctx(priv(), priv())) == (0.0, {})


def test_from_config_accepts_reward_block(reg) -> None:
    comp = RewardComposer.from_config({"terms": [{"name": "fake_const", "weight": 2.0}]})
    start(comp, priv())
    assert comp(ctx(priv(), priv()))[0] == 2.0
    with pytest.raises(ValueError):
        RewardComposer.from_config({"terms": [], "typo": 1})


# ---------------------------------------------------------------------- reset per episode and episode stats
def test_reset_per_episode_and_episode_stats(reg) -> None:
    comp = RewardComposer([
        {"name": "fake_const", "weight": 2.0, "params": {"value": 1.5}},
        {"name": "fake_key_first", "weight": 3.0},
    ])
    const = comp._entries[0].term
    p0, p1 = priv(), priv(keys=(1, 0, 0, 0, 0, 0))
    start(comp, p0)
    assert const.resets == 1
    comp(ctx(p0, p1, 0))
    comp(ctx(p1, p1, 1))
    s = comp.episode_stats()
    assert s.steps == 2
    assert s.raw == {"fake_const": 3.0, "fake_key_first": 1.0}
    assert s.weighted == {"fake_const": 6.0, "fake_key_first": 3.0}
    assert s.nonzero_steps == {"fake_const": 2, "fake_key_first": 1}
    assert s.total == pytest.approx(9.0)
    start(comp, p0)  # next episode
    assert const.resets == 2
    s = comp.episode_stats()
    assert s.steps == 0 and s.total == 0.0 and set(s.raw.values()) == {0.0}
    _, terms = comp(ctx(p0, p1, 0))
    assert terms["fake_key_first"] == 1.0  # the ledger was cleared: the same key pays again next episode


def test_call_before_reset_raises(reg) -> None:
    comp = RewardComposer([{"name": "fake_const", "weight": 1.0}])
    with pytest.raises(RuntimeError):
        comp(ctx(priv(), priv()))


# ---------------------------------------------------------------------- D10 first-occurrence semantics
def test_key_pays_only_first_time_per_episode(reg) -> None:
    comp = RewardComposer([{"name": "fake_key_first", "weight": 3.0}])
    none, blue, blue_red = priv(), priv(keys=(1, 0, 0, 0, 0, 0)), priv(keys=(1, 0, 1, 0, 0, 0))
    start(comp, none)
    seq = [(none, blue), (blue, blue), (blue, none), (none, blue), (blue, blue_red), (blue_red, blue_red)]
    got = [comp(ctx(a, b, i))[1]["fake_key_first"] for i, (a, b) in enumerate(seq)]
    assert got == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # dropping and re-gaining blue does not pay twice


def test_door_that_recloses_and_reopens_pays_once(reg) -> None:
    comp = RewardComposer([{"name": "fake_door_first", "weight": 1.0}])
    closed = priv(floor=(0, 0, 0), ceil=(128, 0, 0))  # sector 0 is a room; 1 and 2 are doors
    d1_open = priv(floor=(0, 0, 0), ceil=(128, 72, 0))
    both = priv(floor=(0, 0, 0), ceil=(128, 72, 72))
    start(comp, closed)
    seq = [closed, d1_open, d1_open, closed, d1_open, both, closed, both]
    got = [comp(ctx(seq[max(i - 1, 0)], s, i))[1]["fake_door_first"] for i, s in enumerate(seq)]
    # Step 0 pays for sector 0 (an always-open room) once: real terms filter by inventory["door_sectors"].
    assert got == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_first_occurrence_ledger() -> None:
    led = FirstOccurrence()
    assert led.first("a") and not led.first("a")
    assert led.update(["b", "a", "c", "b"]) == ["b", "c"]
    assert len(led) == 3 and "c" in led
    led.clear()
    assert len(led) == 0 and led.first("a")
    led2 = FirstOccurrence()
    assert new_set_bits(np.array([0, 1, 0, 1]), led2) == [1, 3]
    assert new_set_bits(np.array([1, 1, 0, 1]), led2) == [0]
    assert new_set_bits(np.array([0, 0, 0, 0]), led2) == []


def test_base_term_clears_all_its_ledgers_on_reset() -> None:
    t = DoorFirstTerm()
    extra = t.ledger()
    t.opened.first(1)
    extra.first("x")
    t.reset(priv(), {}, {}, None)
    assert len(t.opened) == 0 and len(extra) == 0


# ---------------------------------------------------------------------- loud failures and registry
def test_non_finite_term_value_raises(reg) -> None:
    comp = RewardComposer([{"name": "fake_nan", "weight": 1.0}])
    start(comp, priv())
    with pytest.raises(ValueError, match="fake_nan"):
        comp(ctx(priv(), priv()))


@pytest.mark.parametrize(
    "cfg, exc",
    [
        ([{"name": "no_such_term", "weight": 1.0}], KeyError),
        ([{"name": "fake_const", "weight": 1.0}, {"name": "fake_const", "weight": 2.0}], ValueError),
        ([{"name": "fake_const"}], ValueError),
        ([{"name": "fake_const", "weight": 1.0, "wieght": 2}], ValueError),
        ([{"name": "fake_const", "weight": float("inf")}], ValueError),
        ([{"name": "fake_const", "weight": 1.0, "params": {"bogus": 1}}], TypeError),
        ({"name": "fake_const", "weight": 1.0}, TypeError),
    ],
    ids=["unknown", "duplicate", "no_weight", "typo_key", "inf_weight", "bad_param", "not_list"],
)
def test_bad_config_raises(reg, cfg, exc) -> None:
    with pytest.raises(exc):
        RewardComposer(cfg)


def test_register_term(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "REWARD_TERMS", {})

    @register_term
    class A(BaseRewardTerm):
        name = "fake_a"

        def value(self, ctx):
            return 0.0

    assert base.REWARD_TERMS == {"fake_a": A}
    assert register_term(A) is A  # idempotent for the same class

    class B(BaseRewardTerm):
        name = "fake_a"

    with pytest.raises(ValueError):
        register_term(B)

    class C(BaseRewardTerm):
        name = "Not-Snake"

    with pytest.raises(ValueError):
        register_term(C)
    assert isinstance(A(), RewardTerm) and isinstance(StepTerm(), RewardTerm)


def test_global_registry_has_no_fakes() -> None:
    assert not any(k.startswith("fake_") for k in REWARD_TERMS)


def test_load_term_modules_skips_only_missing_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "fake_reward_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "ok.py").write_text("LOADED = True\n")
    (pkg / "broken.py").write_text("import definitely_not_a_real_module_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert load_term_modules(["fake_reward_pkg.ok", "fake_reward_pkg.absent"]) == ["fake_reward_pkg.ok"]
    with pytest.raises(ModuleNotFoundError):
        load_term_modules(["fake_reward_pkg.broken"])  # a real import error inside a module propagates


def test_custom_registry_argument() -> None:
    comp = RewardComposer([{"name": "fake_const", "weight": 1.0}], registry={"fake_const": ConstTerm})
    start(comp, priv())
    assert comp(ctx(priv(), priv())) == (1.0, {"fake_const": 1.0})


def test_step_context_validates_action_and_is_frozen() -> None:
    with pytest.raises(ValueError):
        StepContext(priv(), priv(), {}, np.zeros(8, np.int8), False, False, 0.0, {}, 0)
    c = ctx(priv(), priv())
    with pytest.raises(AttributeError):
        c.step = 3  # type: ignore[misc]


def test_rewards_package_import_is_torch_and_env_free() -> None:
    code = (
        "import sys, training.rewards\n"
        "print(','.join(m for m in sys.modules if m.split('.')[0] in ('torch', 'doom_env', 'vizdoom', 'doom_agent')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_submission_never_imports_rewards() -> None:
    for p in (REPO / "submission").rglob("*.py"):
        src = p.read_text()
        assert "training.rewards" not in src and "privileged_state" not in src, p
