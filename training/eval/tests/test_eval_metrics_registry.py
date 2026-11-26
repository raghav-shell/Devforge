"""Eval-owned checks: ALL_METRICS plug-in registry and the D13 inventory shim."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from training.eval import metrics as M
from training.eval.inventory import MapInventory, map_inventory
from training.eval.metrics.core import EpisodeTrace, core_metrics

ROOT = Path(__file__).resolve().parents[3]


def test_all_metrics_is_core_first_list_of_callables():
    assert M.ALL_METRICS and M.ALL_METRICS[0] is core_metrics
    assert all(callable(f) for f in M.ALL_METRICS)
    assert set(M.METRICS_BY_MODULE) == {"core", "exploration", "combat"}


def _inv():
    return MapInventory(num_sectors=1, door_sectors={}, keys_present=0, monsters_present=0, sector_area=np.zeros(1),
                        cells_total=1)


def test_run_all_rejects_bad_prefix_collision_and_non_scalar():
    tr, inv = EpisodeTrace(episode_ticks=10), _inv()
    with pytest.raises(ValueError, match="prefix"):
        M.run_all(tr, inv, tagged=[("exploration", lambda t, i: {"core.x": 1})])
    with pytest.raises(ValueError, match="duplicate"):
        M.run_all(tr, inv, tagged=[("combat", lambda t, i: {"combat.a": 1}), ("combat", lambda t, i: {"combat.a": 2})])
    with pytest.raises(TypeError):
        M.run_all(tr, inv, tagged=[("combat", lambda t, i: {"combat.a": [1]})])
    assert M.run_all(tr, inv, tagged=[("combat", lambda t, i: {"combat.a": 1.5})]) == {"combat.a": 1.5}


def test_missing_plugin_skipped_but_real_import_errors_propagate(monkeypatch):
    assert M._load("does_not_exist_yet") == []
    import importlib

    def boom(name):
        raise ModuleNotFoundError("No module named 'scipy_missing'", name="scipy_missing")

    monkeypatch.setattr(importlib, "import_module", boom)
    with pytest.raises(ModuleNotFoundError):
        M._load("exploration")


def test_inventory_shim_two_rooms():
    """D13: values come from training.envs.inventory; MEASURED reference: two_rooms 4 sectors, 60 cells, 1 door."""
    from doom_env import make_env

    from training.envs.inventory import map_inventory as env_inv

    env = make_env(str(ROOT / "maps" / "two_rooms.wad"), "MAP01", episode_ticks=2100)
    try:
        env.reset(seed=0)
        inv = map_inventory(env)
        d = env_inv(env)
        x, y = env.privileged_state()["x"], env.privileged_state()["y"]
    finally:
        env.close()
    assert isinstance(inv, MapInventory)
    assert (inv.num_sectors, inv.cells_total, inv.doors_total) == (4, 60, 1)
    assert inv.monsters_present == d["monsters"] and inv.keys_present == d["keys"]
    assert np.array_equal(inv.sector_area, d["sector_area"])
    assert 0 <= inv.sector_at(x, y) < inv.num_sectors
