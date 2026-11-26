"""Smoke run on maps/two_rooms.wad with the REAL DoomPolicy (B2) and SubprocVecEnv (A3).

Skipped until both land. This is an integration test, not a unit test: one tiny PPO update on the
real env, then the epoch-1 ratio gate, a checkpoint and an export.
"""

import importlib.util
import json

import pytest
from conftest import ROOT

HAVE = all(importlib.util.find_spec(m) is not None for m in ("doom_agent.models.policy", "training.envs.vector"))
pytestmark = pytest.mark.skipif(not HAVE, reason="needs B2 (doom_agent.models) and A3 (training.envs.vector)")


def test_two_rooms_smoke(tmp_path):
    from training.configs import TrainConfig
    from training.ppo.export import export_policy
    from training.train import run

    manifest = tmp_path / "two_rooms.json"
    smoke = json.loads((ROOT / "maps/splits/smoke.json").read_text())
    smoke["maps"] = [m for m in smoke["maps"] if m["id"] == "two_rooms"]
    manifest.write_text(json.dumps(smoke))
    cfg = TrainConfig().with_overrides([
        "ppo.num_envs=2", "ppo.rollout_len=32", "ppo.num_minibatches=1", "ppo.epochs=1",
        "ppo.ratio_check_fail=true", "env.num_workers=1", "env.envs_per_worker=2",
        f'env.manifests=["{manifest}"]', "run.tensorboard=false", "run.torch_threads=1",
        # a small, fast term set for the smoke run (the H1 set incl. cell_visit / kill / damage_taken is in h1.json)
        'reward.terms=[{"name": "exit_reached", "weight": 10.0}, {"name": "door_open", "weight": 1.0}, '
        '{"name": "death", "weight": -3.0}]'])
    rd = run(cfg, max_updates=2, run_dir=tmp_path / "run")
    recs = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    assert len(recs) == 2
    for rec in recs:
        assert rec["ppo/ratio_check_max_dev"] < 1e-5
        assert rec["env/crashed_total"] == 0
    meta = json.loads((rd / "run.json").read_text())
    assert meta["fake"] is False and meta["action_api_source"] == "doom_agent.models"
    assert meta["train_manifest_ids"] == ["two_rooms"]
    ck = rd / "checkpoints" / "ckpt_000002.pt"
    assert ck.exists()
    # real resume: learner state restored, episodes restart (SubprocVecEnv cannot be snapshotted)
    run(cfg, resume=str(rd), max_updates=1)
    recs = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    assert recs[-1]["update"] == 3 and recs[-1]["ppo/ratio_check_max_dev"] < 1e-5
    meta = export_policy(rd / "checkpoints" / "ckpt_000003.pt", tmp_path / "export")
    assert meta["global_step"] == 3 * 2 * 32
    # the export loads and plays through a copy of submission/agent.py (strict mode), not submission/weights/
    from training.ppo.agent_check import check

    res = check(tmp_path / "export", "maps/two_rooms.wad", steps=40)
    assert res["steps"] == 40 and res["weights_dir"].startswith(str(tmp_path))
