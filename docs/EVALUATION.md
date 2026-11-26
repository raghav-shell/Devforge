# How Evaluation Works

> Exact weights and map details are kept hidden. This page explains *what* is rewarded.
> Values marked *(draft)* are announced before the event.

## Hardware

One NVIDIA RTX 3090 (24 GB) shared by evaluation workers. Each agent gets:

- **2 GB VRAM**. Enforced for PyTorch (`torch.cuda.set_per_process_memory_fraction`, applied
  right after your `__init__` returns) and JAX (`XLA_PYTHON_CLIENT_MEM_FRACTION`); allocating beyond
  it raises a CUDA out-of-memory error.
- **8 GB RAM**, including anything you write to `/tmp` or `/dev/shm` (they are in-memory). Going
  over it kills your agent (counts as a crash).
- **1 CPU thread**: your agent gets one CPU's worth of time. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS` are set to 1 and JAX/XLA runs single-threaded on the CPU; extra threads
  only compete for the same CPU. Do heavy compute on the GPU, or keep CPU models small.
- No network; repository mounted read-only; writable `/tmp` (2 GB)

## Which code is evaluated

The last commit on your repository's default branch **at the submission deadline**. Organizers
record that commit for every team when the deadline passes; later pushes are ignored.

## Hidden map suite

All maps are **unseen**. Every agent plays every map, in order of difficulty, with fixed seeds.

| Tier | Category | Skills tested |
| --- | --- | --- |
| 1 | Combat & Survival | Aiming, dodging projectiles, managing health/ammo |
| 2 | Multi-Room Exploration | Partial observability, spatial memory, avoiding dead-end traps, doors |
| 3 | Long-Horizon Progression | Finding keycards, remembering locked doors, backtracking to the exit |
| 4 | Full Gauntlet | All of the above with ambushes, scarce resources and multiple locks |

The test map in `maps/` is **not** an evaluation map.

## Scoring

Scores are **not binary**. Partial progress always counts. Each map awards points for:

| Category | Rewarded for |
| --- | --- |
| **Exit reached** (largest) | Reaching the level exit, plus a small efficiency bonus for remaining health and time |
| **Progression** | Picking up keycards, opening doors, unlocking locked doors |
| **Exploration** | Unique areas of the map visited (spinning in place earns nothing) |
| **Combat & survival** | Enemies killed; small penalties for self-damage and dying early |

Harder tiers carry more weight. Your leaderboard score is the weighted sum over all maps.

## Timeouts

| Limit | Value *(draft)* | What happens when it's exceeded |
| --- | --- | --- |
| Initialization (`__init__`, weights onto GPU) | 60 s | 0 points for the affected map |
| `reset()` before each map | 10 s | 0 points for the affected map |
| First `act()` call per map (JIT warm-up) | 10 s | Your agent stops (see below) |
| Any other `act()` call | 5 s | Your agent stops |
| Agent compute time per map | 40 ms × decision steps in the map's tick cap (e.g. 21 s for a 2,100-tick map) | Your agent stops |

These values live in `env/doom_env/limits.py`, and `validate.py` enforces exactly the same numbers.

Only time spent waiting for your agent counts toward the compute budget, not the game engine.

**When your agent stops** (exception, invalid action, timeout, crash), your player **stands still**
for the rest of that map, and the game keeps running until the tick cap, death or exit. You keep the
points earned so far, and penalties such as dying still apply. The agent is restarted for the next map.

## Replays

The scorer records a replay of every run. Organizers watch the top teams' replays during manual review.
