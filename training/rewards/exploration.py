"""Exploration reward terms. TRAINING-ONLY (reads true x, y from `privileged_state()`).

v1 (docs/DECISIONS.md D8, docs/MODULE_CONTRACTS.md §0 and §6.2): the episodic true-cell count bonus,
paid in r_ext through the reward composer. One value head; there is no intrinsic stream in v1.

    {name: cell_visit, weight: 0.05, params: {cell: 64}}

Definition (identical to the eval coverage metric, training/eval/metrics/core.py and
docs/EVALUATION_PROTOCOL.md §4.3, so the shaped signal and the scored signal agree):
- cell c(p) = (floor(x / cell), floor(y / cell)) with cell = 64 map units;
- the cell is taken at the END-of-step position only (one sample per decision, no segment
  interpolation), exactly like the eval trace, which records one row per decision;
- the spawn cell c(p_0) is pre-recorded at reset, so it never pays;
- each other cell pays +1 the first time it is entered this episode and never again (D10), so
  re-entering a cell, oscillating across a cell border, spinning or standing still all pay 0;
- the ledger is cleared on every reset (BaseRewardTerm), so each episode starts fresh.

Hence the per-episode sum of the UNWEIGHTED term equals `core.cells_new` for the same trajectory
(tested in tests/exploration/test_cell_visit.py on a real engine walk).

Nothing under `submission/` may import this module.
"""

from __future__ import annotations

import math
from typing import Any

from .base import BaseRewardTerm, StepContext, register_term

# Must equal training.envs.inventory.CELL (the eval coverage cell). Not imported, so a transient
# breakage in training/envs cannot stop this term from registering; the equality is a test.
CELL = 64.0


def cell_of(x: float, y: float, cell: float = CELL) -> tuple[int, int]:
    """(floor(x / cell), floor(y / cell)). `floor`, not `int`, so cells do not merge across 0."""
    return math.floor(x / cell), math.floor(y / cell)


@register_term
class CellVisit(BaseRewardTerm):
    """+1 for each true-position cell entered for the first time this episode (spawn cell excluded)."""

    name = "cell_visit"

    def __init__(self, cell: float = CELL) -> None:
        super().__init__()
        try:
            c = float(cell)
        except (TypeError, ValueError) as e:
            raise TypeError(f"cell_visit: cell must be a number, got {cell!r}") from e
        if not math.isfinite(c) or c <= 0:
            raise ValueError(f"cell_visit: cell must be finite and > 0, got {cell!r}")
        self.cell = c
        self._visited = self.ledger()  # cleared by BaseRewardTerm.reset at every episode start

    def on_reset(self, priv0: dict, obs0: dict, inventory: dict, spec: Any) -> None:
        self._visited.first(cell_of(float(priv0["x"]), float(priv0["y"]), self.cell))  # spawn never pays

    def value(self, ctx: StepContext) -> float:
        x, y = float(ctx.priv["x"]), float(ctx.priv["y"])
        if not (math.isfinite(x) and math.isfinite(y)):
            return 0.0
        return 1.0 if self._visited.first(cell_of(x, y, self.cell)) else 0.0

    @property
    def cells_visited(self) -> int:
        """Distinct cells this episode, spawn included (= core.cells_visited)."""
        return len(self._visited)
