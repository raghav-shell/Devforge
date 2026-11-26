"""Official evaluation limits, shared by validate.py and the scorer (see docs/EVALUATION.md)."""

MAX_REPO_BYTES = 1024**3  # 1 GB, including LFS objects
INIT_TIMEOUT_S = 60.0  # DoomAgent.__init__ (load weights onto the device)
RESET_TIMEOUT_S = 10.0  # DoomAgent.reset
FIRST_ACT_TIMEOUT_S = 10.0  # first act() of every map (JIT / CUDA warm-up)
ACT_TIMEOUT_S = 5.0  # any other act()
AGENT_MS_PER_STEP = 40.0  # per-map agent compute budget = this x decision steps in the map's tick cap
# e.g. a 2100-tick map has 525 decision steps -> 21 s of agent time; a 4200-tick map -> 42 s.
# Only time spent waiting for your agent counts (not the game engine).


def agent_budget_s(episode_ticks: int, ms_per_step: float = AGENT_MS_PER_STEP) -> float:
    from math import ceil

    from .spaces import FRAME_SKIP

    return ms_per_step / 1000.0 * ceil(episode_ticks / FRAME_SKIP)
MAX_VRAM_MB = 2048  # per agent; enforced for PyTorch and JAX by the agent server / scorer
MAX_RAM_MB = 8192  # per agent: container memory limit (includes files you write to /tmp and /dev/shm)
CPU_THREADS = 1  # per agent: one CPU's worth of time (docker --cpus 1)

# Set in the agent's environment so numerical libraries start one compute thread instead of one per
# host core (with a 1-CPU quota, extra threads only add contention). Read when the library loads.
THREAD_ENV = {
    "OMP_NUM_THREADS": str(CPU_THREADS),
    "MKL_NUM_THREADS": str(CPU_THREADS),
    "OPENBLAS_NUM_THREADS": str(CPU_THREADS),
    "NUMEXPR_NUM_THREADS": str(CPU_THREADS),
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}

# submission/requirements.txt: one `name[extras] <specifiers> [; marker]` per line. No options
# (-r, -e, --index-url, --find-links, ...), URLs, paths or `pkg @ url` references. Max 50 lines.
REQUIREMENT_PATTERN = (
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?"
    r"(\s*(==|!=|<=|>=|~=|<|>)\s*[A-Za-z0-9.*+!_-]+)*"
    r"(\s*,\s*(==|!=|<=|>=|~=|<|>)\s*[A-Za-z0-9.*+!_-]+)*"
    r"(\s*;\s*[A-Za-z0-9_.<>=!~\"' ]+)?$"
)
MAX_REQUIREMENTS = 50
