"""DoomPolicy benchmarks: param count, 1-thread batch-1 latency, training-minibatch throughput (CPU vs MPS),
memory, CPU/MPS numerical drift.

Usage:
    .venv/bin/python training/benchmarks/model_latency.py [--iters 2000] [--json OUT.json] [--quick]

Owner: model-architect. Not imported by anything; standalone script.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "submission", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from doom_agent.models.config import PolicyConfig  # noqa: E402
from doom_agent.models.distributions import select_action  # noqa: E402
from doom_agent.models.io import features_to_torch  # noqa: E402
from doom_agent.models.policy import DoomPolicy  # noqa: E402
from doom_agent.obs.schema import dummy_features  # noqa: E402

VARIANTS = {
    "v1_default": {},
    "film": {"fusion": "film"},
    "mlp_rays": {"ray_encoder": "mlp"},
    "attnpool": {"entity_encoder": "attnpool"},
    "bernoulli": {"action_head": "bernoulli"},
}


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024  # bytes on macOS, KiB on Linux


def bench_step_latency(cfg: PolicyConfig, iters: int, warmup: int = 200) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    pol = DoomPolicy(cfg).eval()
    rng = np.random.default_rng(0)
    pool = [dummy_features((), rng) for _ in range(64)]  # vary inputs; 1-env, no batch dim
    srng = np.random.default_rng([0, 1])
    state = pol.initial_state(1)
    start = torch.tensor([True])
    bare, full = [], []
    with torch.inference_mode():
        xs = [features_to_torch({k: v[None] for k, v in f.items()}, "cpu") for f in pool]
        for i in range(warmup):
            out, state = pol.step(xs[i % 64], state, start)
        for i in range(iters):
            x = xs[i % 64]
            t0 = time.perf_counter()
            out, state = pol.step(x, state, start)
            bare.append(time.perf_counter() - t0)
        for i in range(iters):
            f = pool[i % 64]
            t0 = time.perf_counter()  # the DoomAgent.act path minus ObsPreprocessor
            x = features_to_torch({k: v[None] for k, v in f.items()}, "cpu")
            out, state = pol.step(x, state, start)
            a = select_action(out["logits"][0].numpy(), "sample", 1.0, srng, cfg.action_head)
            full.append(time.perf_counter() - t0)
    assert a.shape == (9,)
    ms = lambda xs: {"p50_ms": pct(xs, 0.5) * 1e3, "p99_ms": pct(xs, 0.99) * 1e3, "mean_ms": statistics.fmean(xs) * 1e3}
    return {"params": pol.num_params(), "step": ms(bare), "act_path": ms(full)}


def _minibatch(T: int, B: int, device: str, seed: int = 0):
    x = features_to_torch(dummy_features((T, B), np.random.default_rng(seed)), device)
    g = torch.Generator().manual_seed(seed)
    es = torch.rand(T, B, generator=g) < 0.01
    es[0] = True
    return x, es.to(device)


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def bench_train_throughput(device: str, threads: int, T: int, B: int, reps: int) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    pol = DoomPolicy(PolicyConfig()).to(device)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4, eps=1e-5)
    x, es = _minibatch(T, B, device)
    s0 = pol.initial_state(B, device)
    act = torch.zeros(T, B, 6, dtype=torch.int64, device=device)

    def fwd_bwd():
        out, _ = pol.forward_sequence(x, s0, es)
        d = pol.distribution(out["logits"])
        loss = -d.log_prob(act).mean() - 0.005 * d.entropy().mean() + 0.5 * out["value"].square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def fwd_only():
        with torch.no_grad():
            pol.forward_sequence(x, s0, es)

    res = {}
    for name, fn in (("fwd_bwd_adam", fwd_bwd), ("fwd_nograd", fwd_only)):
        for _ in range(3):
            fn()
        _sync(device)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            _sync(device)
            ts.append(time.perf_counter() - t0)
        med = statistics.median(ts)
        res[name] = {"median_ms": med * 1e3, "min_ms": min(ts) * 1e3, "samples_per_s": T * B / med}
    if device == "mps":
        res["mps_current_allocated_mb"] = torch.mps.current_allocated_memory() / 2**20
        res["mps_driver_allocated_mb"] = torch.mps.driver_allocated_memory() / 2**20
    return res


def bench_rollout_step(threads: int, B: int, iters: int = 300) -> dict:
    """Batched no-grad step as used during rollout collection (N envs, CPU)."""
    torch.set_num_threads(threads)
    pol = DoomPolicy(PolicyConfig()).eval()
    x = features_to_torch(dummy_features(B), "cpu")
    s = pol.initial_state(B)
    es = torch.zeros(B, dtype=torch.bool)
    ts = []
    with torch.inference_mode():
        for _ in range(30):
            pol.step(x, s, es)
        for _ in range(iters):
            t0 = time.perf_counter()
            pol.step(x, s, es)
            ts.append(time.perf_counter() - t0)
    return {"p50_ms": pct(ts, 0.5) * 1e3, "p99_ms": pct(ts, 0.99) * 1e3}


def mps_drift(T: int, B: int) -> float:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    pol = DoomPolicy(PolicyConfig())
    with torch.no_grad():
        for prm in pol.parameters():  # leave gain-0.01 init regime so drift is visible in logits
            prm.add_(0.05 * torch.randn_like(prm))
    x, es = _minibatch(T, B, "cpu")
    with torch.no_grad():
        a, _ = pol.forward_sequence(x, pol.initial_state(B), es)
        pm = pol.to("mps")
        xm = {k: v.to("mps") for k, v in x.items()}
        b, _ = pm.forward_sequence(xm, pm.initial_state(B, "mps"), es.to("mps"))
    return float((a["logits"] - b["logits"].cpu()).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="v1 only, fewer iterations")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    if args.quick:
        args.iters, args.reps = 300, 3

    default_threads = torch.get_num_threads()
    report: dict = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
        "default_threads": default_threads,
        "rss_start_mb": rss_mb(),
        "latency_1thread_batch1": {},
    }
    variants = {"v1_default": {}} if args.quick else VARIANTS
    for name, kw in variants.items():
        r = bench_step_latency(PolicyConfig(**kw), args.iters)
        report["latency_1thread_batch1"][name] = r
        print(f"[latency] {name:11s} params={r['params']:,}  step p50={r['step']['p50_ms']:.3f} "
              f"p99={r['step']['p99_ms']:.3f} ms | act_path p50={r['act_path']['p50_ms']:.3f} "
              f"p99={r['act_path']['p99_ms']:.3f} ms", flush=True)
    report["rss_after_latency_mb"] = rss_mb()

    report["rollout_step_cpu"] = {}
    for thr in (1, default_threads):
        for B in (16, 64):
            r = bench_rollout_step(thr, B)
            report["rollout_step_cpu"][f"threads{thr}_B{B}"] = r
            print(f"[rollout] cpu threads={thr} B={B}: p50={r['p50_ms']:.3f} p99={r['p99_ms']:.3f} ms", flush=True)

    report["train_minibatch"] = {"T": args.T, "B": args.B}
    for thr in (1, default_threads):
        r = bench_train_throughput("cpu", thr, args.T, args.B, args.reps)
        report["train_minibatch"][f"cpu_threads{thr}"] = r
        print(f"[train] cpu threads={thr}: fwd+bwd+adam {r['fwd_bwd_adam']['median_ms']:.1f} ms "
              f"({r['fwd_bwd_adam']['samples_per_s']:.0f} samples/s); fwd {r['fwd_nograd']['median_ms']:.1f} ms",
              flush=True)
    report["rss_after_cpu_train_mb"] = rss_mb()
    if torch.backends.mps.is_available():
        try:
            r = bench_train_throughput("mps", default_threads, args.T, args.B, args.reps)
            report["train_minibatch"]["mps"] = r
            print(f"[train] mps: fwd+bwd+adam {r['fwd_bwd_adam']['median_ms']:.1f} ms "
                  f"({r['fwd_bwd_adam']['samples_per_s']:.0f} samples/s); fwd {r['fwd_nograd']['median_ms']:.1f} ms; "
                  f"mps alloc {r['mps_current_allocated_mb']:.1f} MB (driver {r['mps_driver_allocated_mb']:.1f} MB)",
                  flush=True)
            report["mps_vs_cpu_max_abs_logit_diff"] = mps_drift(args.T, args.B)
            print(f"[drift] max |logits_cpu - logits_mps| = {report['mps_vs_cpu_max_abs_logit_diff']:.3e}")
        except Exception as e:  # report, don't hide
            report["train_minibatch"]["mps_error"] = repr(e)
            print(f"[train] mps FAILED: {e!r}")
    else:
        report["train_minibatch"]["mps"] = "unavailable"
    report["rss_peak_mb"] = rss_mb()
    print(f"[memory] peak RSS {report['rss_peak_mb']:.0f} MB (start {report['rss_start_mb']:.0f} MB)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
