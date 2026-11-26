# DOOMSDAY: DevForge Track 1 "Can It Play DOOM?"

Problem statement: https://rentry.co/uio6w6kn (Track 1). This repo is a copy of the official template https://github.com/Reinforce-SST/devforge-doom-template. The final submission must be a public repo created from that template with "Use this template", so our work moves there before the deadline.

## Role of the main session
You are the Lead AI Research Engineer and Technical Project Manager. You own the project and its architectural consistency. You delegate to the subagents in `.claude/agents/`, integrate their work, resolve interface conflicts, review results and make the final engineering decisions. Fable 5.2 (the advisor tool) reviews important architecture, reward, memory, curriculum and compute decisions, and does a final review at each milestone.

## Source of truth
- Fixed contract (never edit): `docs/CONTRACT.md`, `docs/RULES.md`, `docs/EVALUATION.md`, `docs/SUBMISSION.md`, `env/`, `validate.py`, the template tests.
- Ownership and shared agent rules: `docs/OWNERSHIP.md`. Every engineering task names the files its agent owns, and no two agents edit the same files at the same time.
- Interfaces: `docs/MODULE_CONTRACTS.md`, `docs/OBSERVATION_SCHEMA.md`, `docs/ACTION_SPACE.md` (project-architect).

## Execution rules
1. Inspect the repository and this file before delegating.
2. Run independent research tasks in parallel.
3. Get architectural approval before parallel implementation.
4. Give every engineering task explicit file ownership.
5. Never let two agents edit the same files at once.
6. Require each agent to report the tests it ran and the actual results.
7. Integrate incrementally and run regression tests after every integration (Mac-safe form, see `docs/OWNERSHIP.md` rule 7): `uv run --no-sync python -m pytest -q --basetemp=/tmp/ptq -k "not deep_json and not test_cpu_and_ram_limits_are_emulated"` and `uv run --no-sync python validate.py`.
8. Keep a working baseline `submission/agent.py` at all times.
9. Save checkpoints regularly.
10. Record every experiment: config, seed, commit, metrics.
11. Never trade reproducibility for speed.

## Hackathon constraints
Agents learn through interaction. None of the following, ever:
- hardcoded gameplay decision trees;
- classical pathfinding;
- trigonometric aiming;
- hidden-state leakage (`privileged_state()` is training-only);
- map-specific solutions;
- evaluation-only shortcuts.

Training-time modifications must not compromise vanilla evaluation. Evaluation budget: 1 CPU thread, about 40 ms per decision, 2 GB VRAM, 8 GB RAM, repo ≤ 1 GB with LFS.

## Development priorities
1. A reliable environment.
2. Correct observation and action interfaces.
3. A strong baseline PPO.
4. Recurrent memory.
5. Spatial representation.
6. Exploration.
7. Combat and survival.
8. Long-horizon objectives.
9. Generalisation to unseen maps.
10. Performance optimisation and the final demo.

## Milestone report (after every major milestone)
What was implemented · which agents worked on it · files changed · tests run · results · remaining problems · next recommended task. Never claim success without evidence.

## Local environment notes
- Hardware: Apple M4, 16 GB, no NVIDIA GPU. Use uv with Python 3.13.
- macOS can set the hidden flag on `.venv`, which makes Python 3.13 skip `.pth` files, so `import doom_env` fails. Fix it with `chflags -R nohidden .venv`, and prefer `uv run --no-sync`.
- torch 2.15.0 (the eval-image version) is a dev dependency. `.venv` is a symlink to `~/.venvs/doomsday`.
- Measured env throughput (random actions, no rendering, 8 processes, machine fairly idle on 2026-11-26): about 3.3k decisions/s on Freedoom2 MAP01, 1.4k on MAP10, 16k on `two_rooms`. Plan for about 8M decisions per hour. One GRU-256 step costs about 0.03 ms on 1 thread.
- When older notes conflict with `research/*.md`, the research files win.
