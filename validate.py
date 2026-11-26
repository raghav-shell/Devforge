#!/usr/bin/env python
"""Local submission validator. Run before submitting:

    python validate.py              # in your local environment
    python validate.py --docker     # inside the official image, sandboxed like the scorer (recommended)

It runs the same checks and limits as the official scorer:
  1. Repository size (tracked files incl. LFS objects) <= 1 GB; large non-LFS files; untracked weights.
  2. submission/agent.py imports and DoomAgent has the required methods.
  3. Full episodes (default: maps/two_rooms.wad, or your own with --map) through the same process
     boundary as the scorer.
  4. Init time, per-step latency and per-map agent compute budget against the official limits.
  5. Peak GPU memory of the agent process (where nvidia-smi can report it).
Exit code 0 = all required checks passed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OFFICIAL_IMAGE = os.environ.get("DOOM_IMAGE", "ghcr.io/reinforce-sst/devforge-doom:latest")
LOCAL_IMAGE = "devforge-doom"  # `docker build -t devforge-doom .` from this repo
DEFAULT_MAP = "maps/two_rooms.wad"
DEFAULT_TICKS = 2100
LARGE_FILE_BYTES = 50 * 1024**2
GITHUB_MAX_FILE_BYTES = 100 * 1024**2

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def report(status: str, check: str, detail: str = "") -> None:
    results.append((status, check, detail))
    mark = {OK: "\033[32m✓\033[0m", WARN: "\033[33m!\033[0m", FAIL: "\033[31m✗\033[0m"}[status]
    print(f" {mark} {check}" + (f": {detail}" if detail else ""), flush=True)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# --------------------------------------------------------------------------- 1. size
def git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.decode("utf-8", "replace") if out.returncode == 0 else None


def check_size(repo: Path, max_bytes: int) -> None:
    files_out = git(repo, "ls-files", "-z")
    if files_out is None:
        report(WARN, "Repository size", "not a git repository; measuring all files (excluding .venv)")
        files = [p for p in repo.rglob("*") if p.is_file() and ".venv" not in p.parts and ".git" not in p.parts]
        rel = [str(p.relative_to(repo)) for p in files]
    else:
        rel = [f for f in files_out.split("\0") if f]
    total = 0
    large_non_lfs = []
    lfs_attr = git(repo, "check-attr", "filter", "--", *rel[:5000]) or "" if rel else ""
    lfs_files = {ln.split(": filter: ")[0] for ln in lfs_attr.splitlines() if ln.endswith(": filter: lfs")}
    for f in rel:
        p = repo / f
        if not p.is_file():
            continue
        size = p.stat().st_size
        total += size
        if size > LARGE_FILE_BYTES and f not in lfs_files:
            large_non_lfs.append((f, size))
        if f in lfs_files and size < 200 and p.read_bytes().startswith(b"version https://git-lfs"):
            report(WARN, "Git LFS", f"{f} is an LFS pointer locally (run `git lfs pull`); size not counted")
    status = OK if total <= max_bytes else FAIL
    report(status, "Repository size (tracked files)", f"{human(total)} of {human(max_bytes)}")
    if large_non_lfs:
        too_big = [f for f, size in large_non_lfs if size > GITHUB_MAX_FILE_BYTES]
        names = ", ".join(f"{f} ({human(size)})" for f, size in large_non_lfs[:5])
        if too_big:
            report(FAIL, "Large files not in Git LFS", f"{names}. GitHub rejects files > 100 MB; track them with Git LFS")
        else:
            report(WARN, "Large files not in Git LFS", f"{names}. Use Git LFS for weights")
    git_dir = repo / ".git"
    if git_dir.is_dir():
        gsize = sum(p.stat().st_size for p in git_dir.rglob("*") if p.is_file())
        if gsize > max_bytes:
            report(WARN, ".git history size", f"{human(gsize)}; the scorer shallow-clones, but pushing/cloning will be slow")
    if files_out is not None:
        check_committed(repo)


def _not_cache(path: str) -> bool:
    return "__pycache__" not in path and not path.endswith((".pyc", ".pyo"))


def check_committed(repo: Path) -> None:
    """The scorer evaluates your last pushed commit, not your working tree. This dry run uses the
    working tree, so flag everything that differs: files git ignores or doesn't track, and edits."""
    ignored = [f for f in (git(repo, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--", "submission") or "").split("\0")
               if f and _not_cache(f)]
    if ignored:
        report(FAIL, "Files in submission/ ignored by .gitignore",
               ", ".join(ignored[:5]) + " (they will be missing in the scorer; fix .gitignore or `git add -f`)")
    untracked = [f for f in (git(repo, "ls-files", "-z", "--others", "--exclude-standard", "--", "submission") or "").split("\0")
                 if f and _not_cache(f)]
    if untracked:
        report(WARN, "Untracked files in submission/", ", ".join(untracked[:5]) + " (not part of your submission until committed)")
    changed = [f for f in (git(repo, "diff", "--name-only", "-z", "HEAD", "--", "submission") or "").split("\0") if f]
    if changed:
        report(WARN, "Uncommitted changes in submission/", ", ".join(changed[:5]) + " (the scorer uses your last pushed commit)")


def check_requirements(repo: Path) -> None:
    req = repo / "submission" / "requirements.txt"
    if not req.is_file():
        return
    import re

    try:
        from doom_env.limits import MAX_REQUIREMENTS, REQUIREMENT_PATTERN
    except ImportError:  # --docker on a host without doom_env: checked again inside the container
        return
    lines = [ln.split("#")[0].strip() for ln in req.read_text(errors="replace").splitlines()]
    lines = [ln for ln in lines if ln]
    bad = [ln for ln in lines if not re.match(REQUIREMENT_PATTERN, ln)]
    unpinned = [ln for ln in lines if "==" not in ln]
    if bad:
        report(FAIL, "requirements.txt", f"unsupported line(s) {bad[:3]}: only `package==version` entries "
               "(no options, URLs, paths or -r/-e)")
    elif len(lines) > MAX_REQUIREMENTS:
        report(FAIL, "requirements.txt", f"{len(lines)} packages (max {MAX_REQUIREMENTS})")
    elif unpinned:
        report(WARN, "requirements.txt", f"pin exact versions (pkg==x.y.z): {', '.join(unpinned[:5])}")
    else:
        report(OK, "requirements.txt", f"{len(lines)} extra package(s)" if lines else "no extra packages")


def _norm(name: str) -> str:
    import re

    return re.sub(r"[-_.]+", "-", name).lower()


def check_imports(repo: Path) -> None:
    """Warn about packages that submission/ imports and that are installed here, but are neither in
    the official image (docker/requirements.lock) nor in submission/requirements.txt. A common slip:
    `uv add something` for local work, then the scorer's environment doesn't have it."""
    import ast
    import importlib.metadata as md
    import re

    lock = ROOT / "docker" / "requirements.lock"
    if not lock.is_file():
        return
    allowed = {_norm(m.group(1)) for m in re.finditer(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==", lock.read_text(), re.M)}
    allowed |= {"doom-env", "vizdoom", "pip", "setuptools"}
    req = repo / "submission" / "requirements.txt"
    if req.is_file():
        for ln in req.read_text(errors="replace").splitlines():
            m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", ln.split("#")[0])
            if m:
                allowed.add(_norm(m.group(1)))
    local = {p.stem if p.suffix == ".py" else p.name for base in (repo / "submission", repo) if base.is_dir()
             for p in base.iterdir() if p.suffix == ".py" or p.is_dir()}
    modules: set[str] = set()
    for path in (repo / "submission").rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue  # the interface check reports it
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    dists = md.packages_distributions()
    missing = {}
    for mod in sorted(modules - set(sys.stdlib_module_names) - local):
        names = dists.get(mod)
        if names and not any(_norm(d) in allowed for d in names):
            missing[mod] = names[0]
    if missing:
        listed = ", ".join(f"{m} ({d}=={md.version(d)})" for m, d in missing.items())
        report(WARN, "Imports", f"not in the official image or requirements.txt: {listed}. "
               "Add them to submission/requirements.txt or your agent won't start at evaluation")


# --------------------------------------------------------------------------- 2. interface
def check_interface(repo: Path) -> bool:
    path = repo / "submission" / "agent.py"
    if not path.is_file():
        report(FAIL, "Entry point", "submission/agent.py not found")
        return False
    code = (
        "import importlib.util, inspect, json, sys\n"
        f"sys.path[:0] = [{str(repo / 'submission')!r}, {str(repo)!r}]\n"
        f"spec = importlib.util.spec_from_file_location('submission_agent', {str(path)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "cls = getattr(m, 'DoomAgent', None)\n"
        "out = {'has_class': inspect.isclass(cls)}\n"
        "if out['has_class']:\n"
        "    # The scorer calls DoomAgent(obs_space, act_space, device), agent.reset(), agent.act(obs) positionally.\n"
        "    for name, args in (('__init__', 3), ('reset', 0), ('act', 1)):\n"
        "        f = getattr(cls, name, None)\n"
        "        if f is None or (name == '__init__' and f is object.__init__):\n"
        "            out[name] = None if f is None else 'object.__init__ takes no arguments'\n"
        "            continue\n"
        "        try:\n"
        "            inspect.signature(f).bind(None, *range(args)); out[name] = 'ok'\n"
        "        except TypeError as e:\n"
        "            out[name] = str(e)\n"
        "        except ValueError:\n"
        "            out[name] = 'ok'  # no signature available (builtin): can't check\n"
        "print('@@' + json.dumps(out))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, cwd=repo)
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")]
    if proc.returncode != 0 or not marker:
        report(FAIL, "Import submission/agent.py", (proc.stderr.strip().splitlines() or ["unknown error"])[-1])
        return False
    info = json.loads(marker[-1][2:])
    if not info["has_class"]:
        report(FAIL, "DoomAgent class", "submission/agent.py does not define a DoomAgent class")
        return False
    ok = True
    for name in ("__init__", "reset", "act"):
        result = info.get(name)
        if result is None:
            report(FAIL, f"DoomAgent.{name}", "missing")
            ok = False
        elif result != "ok":
            want = {"__init__": "(self, observation_space, action_space, device)", "reset": "(self)", "act": "(self, observation)"}[name]
            report(FAIL, f"DoomAgent.{name}", f"can't be called as {want}: {result}")
            ok = False
    if ok:
        report(OK, "DoomAgent interface", "__init__, reset, act")
    return ok


# --------------------------------------------------------------------------- 3-5. dry run
class VramSampler(threading.Thread):
    """Best-effort peak GPU memory of one process via nvidia-smi."""

    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.pid, self.peak, self.supported, self._stop = pid, 0, None, threading.Event()
        # Inside a container nvidia-smi reports host PIDs (or none), so also track whole-GPU usage
        # above the level before the agent started.
        self.baseline = self._gpu_used()
        self.peak_total = 0

    @staticmethod
    def _gpu_used() -> int | None:
        if not shutil.which("nvidia-smi"):
            return None
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.split()
            return sum(int(x) for x in out if x.isdigit()) if out else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    def run(self) -> None:
        if not shutil.which("nvidia-smi"):
            self.supported = False
            return
        while not self._stop.wait(0.5):
            used = self._gpu_used()
            if used is not None and self.baseline is not None:
                self.peak_total = max(self.peak_total, used - self.baseline)
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
            except (OSError, subprocess.TimeoutExpired):
                continue
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit():
                    self.supported = True
                    if int(parts[0]) == self.pid and parts[1].isdigit():
                        self.peak = max(self.peak, int(parts[1]))

    def stop(self) -> None:
        self._stop.set()


class RamSampler(threading.Thread):
    """Peak resident memory (RSS) of the agent process and its children, from /proc (Linux)."""

    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.pid, self.peak_mb, self._stop = pid, 0, threading.Event()
        self.supported = Path(f"/proc/{pid}/status").exists()

    @staticmethod
    def _rss_kb(pid: int) -> int:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        except (OSError, ValueError, IndexError):
            pass
        return 0

    def _tree(self) -> list[int]:
        children: dict[int, list[int]] = {}
        for d in Path("/proc").iterdir():
            if d.name.isdigit():
                try:
                    ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
                except (OSError, ValueError, IndexError):
                    continue
                children.setdefault(ppid, []).append(int(d.name))
        out, todo = [], [self.pid]
        while todo:
            p = todo.pop()
            out.append(p)
            todo += children.get(p, [])
        return out

    def run(self) -> None:
        if not self.supported:
            return
        while not self._stop.wait(0.25):
            self.peak_mb = max(self.peak_mb, sum(self._rss_kb(p) for p in self._tree()) // 1024)

    def stop(self) -> None:
        self._stop.set()


def one_cpu() -> set[int] | None:
    """The CPU the agent is pinned to, like the scorer's 1-CPU limit (None if not possible here)."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    cpus = sorted(os.sched_getaffinity(0))
    return {cpus[-1]} if len(cpus) > 1 else None  # keep the other cores for the game engine


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))] if s else 0.0


def dry_run(repo: Path, device: str, maps: list[str], ticks: int) -> bool:
    from doom_env import limits as L
    from doom_env import make_env, remote

    tmp = Path(tempfile.mkdtemp(prefix="doom-validate-"))
    sock_path = tmp / "agent.sock"
    log_path = tmp / "agent.log"
    server = remote.listen(sock_path)
    log = open(log_path, "wb")
    env_vars = {**os.environ, "PYTHONUNBUFFERED": "1", **L.THREAD_ENV}
    cpu = one_cpu()
    proc = subprocess.Popen(
        [sys.executable, "-m", "doom_env.remote", "serve", "--socket", remote.address(server), "--repo", str(repo), "--device", device],
        stdout=log, stderr=subprocess.STDOUT, env=env_vars, cwd=tmp,
        preexec_fn=(lambda: os.sched_setaffinity(0, cpu)) if cpu else None,
    )
    if cpu:
        report(OK, "CPU limit", f"agent pinned to {L.CPU_THREADS} CPU (core {min(cpu)}), like the evaluation sandbox")
    else:
        report(WARN, "CPU limit", f"could not pin the agent to {L.CPU_THREADS} CPU here; evaluation gives it one CPU, so timings may look better than they will be")
    vram = VramSampler(proc.pid)
    vram.start()
    ram = RamSampler(proc.pid)
    ram.start()
    ok = True
    conn = None
    resolved_device = device
    try:
        t0 = time.perf_counter()
        conn = remote.AgentConnection.accept(server, L.INIT_TIMEOUT_S, alive=lambda: proc.poll() is None)
        ready = conn.init(device, timeout=max(1.0, L.INIT_TIMEOUT_S - (time.perf_counter() - t0)))
        init_s = time.perf_counter() - t0
        resolved_device = str(ready.get("device", device))
        report(OK if init_s <= L.INIT_TIMEOUT_S else FAIL, "Initialization", f"{init_s:.1f}s (limit {L.INIT_TIMEOUT_S:.0f}s) on device '{ready.get('device')}'")
        for spec in maps:
            wad_path, _, map_name = spec.partition(":")
            wad_path = Path(wad_path) if Path(wad_path).is_absolute() else ROOT / wad_path
            name = f"{wad_path.name}:{map_name or 'MAP01'}"
            entry = {"episode_ticks": ticks}
            env = make_env(wad_path, map_name or "MAP01", episode_ticks=ticks)
            try:
                obs, _ = env.reset(seed=0)
                conn.reset(L.RESET_TIMEOUT_S)
                lat: list[float] = []
                budget = L.agent_budget_s(entry["episode_ticks"])
                start = time.perf_counter()
                steps, outcome = 0, "tick cap"
                while True:
                    limit = L.FIRST_ACT_TIMEOUT_S if steps == 0 else L.ACT_TIMEOUT_S
                    remaining = budget - sum(lat)
                    if remaining <= 0:
                        raise remote.AgentFailure("timeout", f"agent compute budget for this map ({budget:.1f}s) used up", "episode")
                    t = time.perf_counter()
                    action = conn.act(obs, timeout=min(limit, remaining))
                    lat.append(time.perf_counter() - t)
                    obs, _, term, trunc, info = env.step(action)
                    steps += 1
                    if term or trunc:
                        outcome = "exit reached" if info["exit_reached"] else ("died" if term else "tick cap")
                        break
                wall = time.perf_counter() - start
                p50, p99, mx = (1000 * percentile(lat, q) for q in (0.5, 0.99, 1.0))
                report(OK, f"Episode on '{name}'", f"{steps} steps, {outcome}, agent time {sum(lat):.1f}s of {budget:.1f}s budget, {wall:.1f}s wall-clock")
                first = 1000 * lat[0]
                rest = lat[1:] or [0.0]
                report(
                    OK if max(rest) <= L.ACT_TIMEOUT_S and lat[0] <= L.FIRST_ACT_TIMEOUT_S else FAIL,
                    f"Step latency '{name}'",
                    f"first {first:.0f} ms, p50 {p50:.1f} ms, p99 {p99:.1f} ms, max {mx:.0f} ms",
                )
                if p50 > 0.6 * L.AGENT_MS_PER_STEP:
                    report(WARN, "Speed margin", f"p50 {p50:.0f} ms/step leaves little headroom: the budget averages {L.AGENT_MS_PER_STEP:.0f} ms per step")
            finally:
                env.close()
    except remote.AgentFailure as e:
        ok = False
        report(FAIL, f"Agent failed ({e.kind} during {e.where})", e.message)
        if e.traceback:
            print("\n".join("     " + ln for ln in e.traceback.strip().splitlines()[-15:]))
    finally:
        if conn is not None:
            conn.close()
        server.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        vram.stop()
        ram.stop()
        log.close()
    if ram.supported and ram.peak_mb:
        status = FAIL if ram.peak_mb > L.MAX_RAM_MB else (WARN if ram.peak_mb > 0.8 * L.MAX_RAM_MB else OK)
        report(status, "Peak RAM", f"{ram.peak_mb} MB resident (limit {L.MAX_RAM_MB} MB, including files written to /tmp)")
    else:
        report(WARN, "Peak RAM", f"could not be measured here (needs Linux /proc); the limit is {L.MAX_RAM_MB} MB")
    if vram.supported and vram.peak:
        report(OK if vram.peak <= L.MAX_VRAM_MB else FAIL, "Peak GPU memory", f"{vram.peak} MB (limit {L.MAX_VRAM_MB} MB)")
    elif vram.peak_total > 0 and resolved_device != "cpu":
        report(OK if vram.peak_total <= L.MAX_VRAM_MB else FAIL, "Peak GPU memory",
               f"~{vram.peak_total} MB above the pre-run level (whole GPU; limit {L.MAX_VRAM_MB} MB)")
    elif resolved_device != "cpu":
        report(WARN, "Peak GPU memory", "could not be measured here (nvidia-smi has no per-process data); try --docker on a Linux GPU host")
    if not ok:
        tail = log_path.read_bytes().decode("utf-8", "replace").strip().splitlines()[-20:]
        if tail:
            print("   last agent output:\n" + "\n".join("     " + ln for ln in tail))
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


# --------------------------------------------------------------------------- docker
def _have_image(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def resolve_image(image: str, explicit: bool) -> str | None:
    """The image to validate in: present locally, else pulled. Without --image, a locally built
    `devforge-doom` is used when the official image can't be found."""
    if _have_image(image):
        return image
    if not explicit and _have_image(LOCAL_IMAGE):
        print(f"(using the locally built image '{LOCAL_IMAGE}'; {image} is not available here)", flush=True)
        return LOCAL_IMAGE
    print(f"$ docker pull {image}", flush=True)
    if subprocess.call(["docker", "pull", image]) == 0:
        return image
    print(f"\nThe image {image} is not available. Build it from this repository instead:\n"
          f"    docker build -t {LOCAL_IMAGE} .\n"
          f"    python validate.py --docker --image {LOCAL_IMAGE}", file=sys.stderr)
    return None


def run_in_docker(repo: Path, image: str, gpus: bool, maps: list[str], ticks: int) -> int:
    """Like the scorer: requirements.txt installed with network into a separate dir, then the agent
    runs with no network, a read-only filesystem and the repo mounted read-only."""
    if not shutil.which("docker"):
        print("docker not found. Install Docker or run without --docker.", file=sys.stderr)
        return 2
    image = resolve_image(image, explicit=image != OFFICIAL_IMAGE)
    if image is None:
        return 2
    work = Path(tempfile.mkdtemp(prefix="doom-validate-docker-"))
    deps = work / "deps"
    deps.mkdir()
    user = ["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
    try:
        req = repo / "submission" / "requirements.txt"
        if req.is_file() and any(ln.split("#")[0].strip() for ln in req.read_text(errors="replace").splitlines()):
            install = ["docker", "run", "--rm", *user, "--read-only", "--tmpfs", "/tmp:size=4g", "-e", "HOME=/tmp",
                       "-v", f"{req}:/req.txt:ro", "-v", f"{deps}:/deps", image,
                       "uv", "pip", "install", "--no-cache", "--python", "/opt/venv/bin/python", "--target", "/deps", "-r", "/req.txt"]
            print("$ " + " ".join(install), flush=True)
            if subprocess.call(install) != 0:
                report(FAIL, "requirements.txt", "installing it in the official image failed (see output above)")
                return 1
        # The agent's RAM and CPU limits are checked inside (peak RSS, pinned to one core). The
        # container also runs this validator and the game engine, hence the extra memory and no --cpus.
        try:
            from doom_env.limits import MAX_RAM_MB
        except ImportError:  # --docker on a host without doom_env installed
            MAX_RAM_MB = 8192
        mem = f"{MAX_RAM_MB + 2048}m"
        cmd = [
            "docker", "run", "--rm", *user, "--network", "none", "--read-only",
            "--memory", mem, "--memory-swap", mem, "--pids-limit", "512",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:exec,size=2g", "--shm-size", "2g",
            "-e", "HOME=/tmp", "-e", "PYTHONPATH=/deps", "-w", "/tmp",
            "-v", f"{repo}:/submission:ro", "-v", f"{deps}:/deps:ro",
            "-v", f"{ROOT / 'validate.py'}:/validator/validate.py:ro",
        ]
        inner_maps = []
        for i, spec in enumerate(maps):
            wad, sep, map_name = spec.partition(":")
            path = Path(wad) if Path(wad).is_absolute() else (Path.cwd() / wad if (Path.cwd() / wad).exists() else ROOT / wad)
            if not path.is_file():
                print(f"map file not found: {path}", file=sys.stderr)
                return 2
            cmd += ["-v", f"{path.resolve()}:/validator/maps/{i}_{path.name}:ro"]
            inner_maps += ["--map", f"/validator/maps/{i}_{path.name}" + (f":{map_name}" if sep else "")]
        if gpus:
            cmd += ["--gpus", "all"]
        cmd += [image, "python", "/validator/validate.py", "--repo", "/submission", "--in-docker",
                "--device", "cuda" if gpus else "cpu", "--ticks", str(ticks), *inner_maps]
        print("$ " + " ".join(cmd), flush=True)
        return subprocess.call(cmd)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(ROOT), help="repository to validate (default: this one)")
    ap.add_argument("--device", default="auto", help="cuda | cpu | auto (default)")
    ap.add_argument("--map", dest="maps", action="append", metavar="WAD[:MAP]",
                    help=f"map(s) to dry-run, e.g. --map my_maps.wad:MAP03 (default: {DEFAULT_MAP})")
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS, help="episode tick cap for the dry run")
    ap.add_argument("--docker", action="store_true", help="run inside the official image, sandboxed like the scorer")
    ap.add_argument("--image", default=OFFICIAL_IMAGE)
    ap.add_argument("--no-gpu", action="store_true", help="with --docker: don't pass --gpus all")
    ap.add_argument("--in-docker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    if os.name == "nt":
        os.system("")  # enables ANSI colours in the classic Windows console

    if args.docker:
        try:
            from doom_env import limits  # noqa: PLC0415 - only needed for the host-side size check

            max_bytes = limits.MAX_REPO_BYTES
        except ImportError:
            max_bytes = 1024**3
        check_size(repo, max_bytes)
        if any(r[0] == FAIL for r in results):
            return 1
        return run_in_docker(repo, args.image, gpus=not args.no_gpu, maps=args.maps or [DEFAULT_MAP], ticks=args.ticks)

    try:
        from doom_env import CONTRACT_VERSION, limits
    except ImportError:
        print("doom_env is not installed. Run `uv sync` (or use --docker).", file=sys.stderr)
        return 2
    print(f"Validating {repo}  (contract {CONTRACT_VERSION})\n", flush=True)
    if not args.in_docker:
        check_size(repo, limits.MAX_REPO_BYTES)
    check_requirements(repo)
    if not args.in_docker:
        check_imports(repo)
    if check_interface(repo):
        dry_run(repo, args.device, args.maps or [DEFAULT_MAP], args.ticks)

    failed = [r for r in results if r[0] == FAIL]
    warned = [r for r in results if r[0] == WARN]
    print()
    if failed:
        print(f"\033[31mFAILED\033[0m: {len(failed)} check(s) failed, {len(warned)} warning(s).")
        return 1
    print(f"\033[32mPASSED\033[0m with {len(warned)} warning(s)." if warned else "\033[32mPASSED\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
