"""numpy feature dict -> torch tensors (docs/MODULE_CONTRACTS.md 3.1).

Leading dims are kept. Known v1 keys are cast to their FEATURE_SPECS dtype (float32 / int64 / bool);
unknown keys keep their dtype. Zero-copy on CPU when the array is already contiguous and correctly typed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from doom_agent.obs.schema import FEATURE_SPECS


def features_to_torch(feats: dict[str, np.ndarray], device: torch.device | str = "cpu") -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for k, v in feats.items():
        if isinstance(v, Tensor):
            t = v
            spec = FEATURE_SPECS.get(k)
            if spec is not None:
                t = t.to(torch.from_numpy(np.zeros(0, dtype=spec.dtype)).dtype)
        else:
            a = np.asarray(v)
            spec = FEATURE_SPECS.get(k)
            if spec is not None and a.dtype != spec.dtype:
                a = a.astype(spec.dtype)
            t = torch.from_numpy(np.ascontiguousarray(a))
        out[k] = t.to(device)
    return out
