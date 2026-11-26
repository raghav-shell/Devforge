"""Metric plug-ins (docs/MODULE_CONTRACTS.md §7, docs/EVALUATION_PROTOCOL.md §9.3).

Each module exports `METRICS: list[MetricFn]` with `MetricFn(EpisodeTrace, MapInventory) -> dict`. Keys carry the
module's prefix; values are JSON scalars. `ALL_METRICS` is core first, then the plug-ins that exist:
- core.py (evaluation researcher), prefix `core.`
- exploration.py (exploration-engineer), prefix `expl.`
- combat.py (combat-and-survival-engineer), prefix `combat.`

A plug-in module that does not exist yet is skipped; any other import error propagates (never swallowed).
`run_all` calls every MetricFn and rejects wrong prefixes and key collisions.
"""

from __future__ import annotations

import importlib

from .core import METRICS as _CORE
from .core import EpisodeTrace, MetricFn  # noqa: F401  re-exported

PLUGINS: dict[str, str] = {"core": "core.", "exploration": "expl.", "combat": "combat."}


def _load(name: str) -> list:
    if name == "core":
        return list(_CORE)
    modname = f"{__name__}.{name}"
    try:
        mod = importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name == modname:
            return []  # plug-in not written yet
        raise
    fns = getattr(mod, "METRICS", None)
    if not isinstance(fns, (list, tuple)) or not all(callable(f) for f in fns):
        raise TypeError(f"{modname}.METRICS must be a list of MetricFn")
    return list(fns)


METRICS_BY_MODULE: dict[str, list] = {name: _load(name) for name in PLUGINS}
_TAGGED: list = [(name, fn) for name, fns in METRICS_BY_MODULE.items() for fn in fns]
ALL_METRICS: list = [fn for _, fn in _TAGGED]  # list[MetricFn], core first (MODULE_CONTRACTS §7)


def run_all(trace, inv, tagged=None) -> dict:
    """Call every registered MetricFn (core first). Raise on a wrong prefix or a duplicate key.
    `tagged`: optional [(module name, MetricFn)] override, for tests."""
    out: dict = {}
    for name, fn in (_TAGGED if tagged is None else tagged):
        prefix = PLUGINS[name]
        res = fn(trace, inv)
        for k, v in res.items():
            if not k.startswith(prefix):
                raise ValueError(f"{name} metric {getattr(fn, '__name__', fn)} returned key {k!r} without prefix {prefix!r}")
            if k in out:
                raise ValueError(f"duplicate metric key {k!r} from {name}")
            if not (v is None or isinstance(v, (bool, int, float, str))):
                raise TypeError(f"metric {k!r} is not a JSON scalar: {type(v).__name__}")
            out[k] = v
    return out
