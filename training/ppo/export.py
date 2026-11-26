"""Weight export, format "export-1" (docs/MODULE_CONTRACTS.md section 8.2).

    python -m training.ppo.export --ckpt runs/.../ckpt_000100.pt --out runs/.../export

Writes:
  DIR/policy.pt    DoomPolicy.state_dict() only, CPU, floating tensors as float32; loadable with
                   torch.load(weights_only=True). No optimiser / aux heads / privileged critic / return norm.
  DIR/policy.json  {format_version: "export-1", policy_config, obs_schema_version, obs_pipeline_hash,
                   obs_config, action_space_version, default_inference, inference_temperature,
                   source_ckpt_sha256, git_commit, global_step}

Hard checks (section 0: these fail hard here even though DoomAgent only warns):
  - checkpoint obs_pipeline_hash == current tree's hash (override flag; every override is logged
    to stderr, recorded in policy.json and appended to DIR/export_overrides.log);
  - obs_schema_version / action_space_version match the current code;
  - exported keys == DoomPolicy(cfg).state_dict() keys, loaded with strict=True;
  - no key contains "aux" or "priv" (aux heads / privileged critic live outside DoomPolicy by design);
  - ObsConfig.enable_* == PolicyConfig.use_* for depth_band / egomap.
The main session copies DIR/* into submission/weights/ and commits via Git LFS.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "submission", REPO_ROOT):  # same import identity as the scorer (MODULE_CONTRACTS 0)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from training.ppo.checkpoint import file_sha256, load_checkpoint  # noqa: E402

FORMAT_VERSION = "export-1"
FORBIDDEN_KEY_PARTS = ("aux", "priv")
FLAG_PAIRS = (("enable_depth_band", "use_depth_band"), ("enable_egomap", "use_egomap"))


class ExportError(RuntimeError):
    pass


def default_policy_factory(policy_config: dict) -> nn.Module:
    try:
        from doom_agent.models.config import PolicyConfig
        from doom_agent.models.policy import DoomPolicy
    except ImportError as e:
        raise ExportError("doom_agent.models (PolicyConfig, DoomPolicy) is not importable; B2 from the "
                          f"model-architect has not landed. Pass policy_factory explicitly. ({e})") from e
    return DoomPolicy(PolicyConfig.from_dict(policy_config))


def _current_action_space_version() -> str | None:
    try:
        from doom_agent.models.action_codec import ACTION_SPACE_VERSION
        return ACTION_SPACE_VERSION
    except ImportError:
        return None


def export_policy(ckpt_path: str | Path, out_dir: str | Path, *,
                  policy_factory: Callable[[dict], nn.Module] | None = None,
                  allow_hash_mismatch: bool = False, default_inference: str = "sample",
                  inference_temperature: float = 1.0) -> dict:
    from doom_agent.obs import OBS_SCHEMA_VERSION, obs_pipeline_hash

    ckpt_path, out_dir = Path(ckpt_path), Path(out_dir)
    ckpt = load_checkpoint(ckpt_path)
    if default_inference not in ("sample", "greedy"):
        raise ExportError(f"default_inference must be 'sample' or 'greedy', got {default_inference!r}")

    notes: list[str] = []
    cur_hash = obs_pipeline_hash()
    if ckpt["obs_pipeline_hash"] != cur_hash:
        msg = (f"obs_pipeline_hash mismatch: checkpoint {ckpt['obs_pipeline_hash'][:16]}... vs current tree "
               f"{cur_hash[:16]}...")
        if not allow_hash_mismatch:
            raise ExportError(msg + " (pass --allow-hash-mismatch for a deliberate migration)")
        notes.append("OVERRIDE " + msg)
        print("WARNING [export]: " + notes[-1], file=sys.stderr)
    if ckpt["obs_schema_version"] != OBS_SCHEMA_VERSION:
        raise ExportError(f"obs_schema_version {ckpt['obs_schema_version']} != current {OBS_SCHEMA_VERSION}")
    asv = _current_action_space_version()
    if asv is not None and ckpt["action_space_version"] != asv:
        raise ExportError(f"action_space_version {ckpt['action_space_version']} != current {asv}")

    pcfg, ocfg = ckpt["policy_config"], ckpt.get("obs_config", {})
    for enable, use in FLAG_PAIRS:
        if bool(ocfg.get(enable, False)) != bool(pcfg.get(use, False)):
            raise ExportError(f"flag authority violated: obs_config.{enable}={ocfg.get(enable)} vs "
                              f"policy_config.{use}={pcfg.get(use)}")

    factory = policy_factory or default_policy_factory
    policy = factory(pcfg)
    expected = set(policy.state_dict().keys())
    state = ckpt["policy_state"]
    got = set(state.keys())
    if got != expected:
        raise ExportError(f"policy_state keys != DoomPolicy(cfg) keys; missing={sorted(expected - got)[:8]} "
                          f"unexpected={sorted(got - expected)[:8]}")
    bad = [k for k in got if any(p in k.lower() for p in FORBIDDEN_KEY_PARTS)]
    if bad:
        raise ExportError(f"policy_state contains aux/privileged keys: {bad[:8]}")
    clean = {k: (v.detach().cpu().to(torch.float32) if v.is_floating_point() else v.detach().cpu()).clone()
             for k, v in state.items()}
    policy.load_state_dict(clean, strict=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(clean, out_dir / "policy.pt")
    meta = {
        "format_version": FORMAT_VERSION,
        "policy_config": pcfg,
        "obs_schema_version": ckpt["obs_schema_version"],
        "obs_pipeline_hash": ckpt["obs_pipeline_hash"],
        "obs_config": ocfg,
        "action_space_version": ckpt["action_space_version"],
        "default_inference": default_inference,
        "inference_temperature": float(inference_temperature),
        "source_ckpt_sha256": file_sha256(ckpt_path),
        "git_commit": ckpt.get("git_commit", "unknown"),
        "global_step": int(ckpt["global_step"]),
    }
    if notes:
        meta["export_overrides"] = notes
        with open(out_dir / "export_overrides.log", "a") as f:
            for n in notes:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {ckpt_path} {n}\n")
    (out_dir / "policy.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    # Re-load exactly the way the agent will.
    reloaded = torch.load(out_dir / "policy.pt", weights_only=True)
    factory(pcfg).load_state_dict(reloaded, strict=True)
    return meta


def _resolve_factory(spec: str | None):
    if not spec:
        return None
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export a ckpt-1 checkpoint to export-1 (policy.pt + policy.json)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-hash-mismatch", action="store_true")
    ap.add_argument("--default-inference", default="sample", choices=["sample", "greedy"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--policy-factory", default=None, help="module:function building the policy from a config dict "
                    "(default: doom_agent.models DoomPolicy)")
    a = ap.parse_args(argv)
    meta = export_policy(a.ckpt, a.out, policy_factory=_resolve_factory(a.policy_factory),
                         allow_hash_mismatch=a.allow_hash_mismatch, default_inference=a.default_inference,
                         inference_temperature=a.temperature)
    print(json.dumps({k: meta[k] for k in ("format_version", "global_step", "obs_pipeline_hash")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ExportError", "FORMAT_VERSION", "default_policy_factory", "export_policy", "main"]
