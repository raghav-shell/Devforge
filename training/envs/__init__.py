"""Training-side environment code (owner: environment-engineer).

Training-only. Nothing here may be imported from `submission/`. Modules:
- `factory.py`: `EnvSpec`, `TrainEnv`, `make_train_env` (official doom_env + ObsPreprocessor + RewardComposer).
- `vector.py`: `SubprocVecEnv`, `StepResult` (one engine per env slot, auto-reset, crash handling).
- `maps.py`: map registry (generated, bundled Freedoom, ViZDoom scenarios -> manifest entries) and `MapSampler`.
  `map_registry.py` is a backward-compatible alias of `maps.py`.
- `mapgen/`: seeded procedural UDMF training-map generator (curriculum stages).
- `smoke_load.py`: load-check every registry entry through the official `doom_env.make_env`.
- `inventory.py`: `map_inventory(env)`, static per-map facts for eval/reward code.
"""
