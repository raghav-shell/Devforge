"""Backward-compatible alias: the registry now lives in `training.envs.maps` (with MapSampler).

    uv run --no-sync python -m training.envs.map_registry ...   # same CLI as `-m training.envs.maps`
"""

import sys

from training.envs import maps as _maps
from training.envs.maps import *  # noqa: F401,F403
from training.envs.maps import (  # noqa: F401  explicit re-exports used by tests and tools
    GENERATED_DIR, ROOT, SOURCES, SPLITS_DIR, MapEntry, all_entries, check_trainable, freedoom1_entries,
    freedoom2_entries, generated_entries, has_player1_start, heldout_keys, main, manifest_entries, resolve_wad, sha256,
    vizdoom_scenario_entries,
)

if __name__ == "__main__":
    sys.exit(_maps.main())
