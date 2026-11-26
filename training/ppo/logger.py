"""Metrics logging: JSONL (always) + TensorBoard (torch.utils.tensorboard, when importable).

`tensorboard` is in the eval/base image but NOT in the local dev venv today. If it is missing, the
logger prints one loud stderr warning and records `tensorboard: "unavailable: <reason>"` so run.json
shows it; JSONL logging is unaffected.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def _clean(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None if math.isnan(v) else (1e308 if v > 0 else -1e308)
    return v


class MetricsLogger:
    def __init__(self, run_dir: str | Path, *, tensorboard: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self.run_dir / "metrics.jsonl", "a", buffering=1)
        self.tb = None
        self.tb_status = "disabled"
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
                self.tb_status = "enabled"
            except Exception as e:  # ImportError when tensorboard is not installed
                self.tb_status = f"unavailable: {type(e).__name__}: {e}"
                print(f"WARNING [MetricsLogger]: TensorBoard logging disabled ({self.tb_status}); "
                      f"JSONL only at {self.run_dir / 'metrics.jsonl'}", file=sys.stderr)

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        rec = {"global_step": int(step)}
        rec.update({k: _clean(float(v)) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
                    for k, v in metrics.items()})
        self._jsonl.write(json.dumps(rec, sort_keys=True) + "\n")
        if self.tb is not None:
            for k, v in metrics.items():
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                if isinstance(v, float) and not math.isfinite(v):
                    continue
                self.tb.add_scalar(k, v, step)

    def close(self) -> None:
        self._jsonl.close()
        if self.tb is not None:
            self.tb.close()


__all__ = ["MetricsLogger"]
