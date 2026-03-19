"""
Reproducibility utilities.

Best-effort seeding across common RNG sources used in this repo:
- Python's `random`
- NumPy (if available)
- PyTorch (if available), including deterministic flag configuration

This module is intentionally dependency-light and safe to import even when
optional libraries are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import random
import sys


@dataclass(frozen=True)
class SeedInfo:
    seed: int
    python_hash_seed: Optional[int]
    random_seeded: bool
    numpy_seeded: bool
    torch_seeded: bool
    torch_deterministic: Optional[bool]
    torch_cudnn_deterministic: Optional[bool]
    torch_cudnn_benchmark: Optional[bool]
    versions: Dict[str, str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "python_hash_seed": self.python_hash_seed,
            "random_seeded": self.random_seeded,
            "numpy_seeded": self.numpy_seeded,
            "torch_seeded": self.torch_seeded,
            "torch_deterministic": self.torch_deterministic,
            "torch_cudnn_deterministic": self.torch_cudnn_deterministic,
            "torch_cudnn_benchmark": self.torch_cudnn_benchmark,
            "versions": dict(self.versions),
        }


def _safe_version(mod_name: str) -> Optional[str]:
    try:
        mod = __import__(mod_name)
        v = getattr(mod, "__version__", None)
        return str(v) if v is not None else None
    except Exception:
        return None


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> Dict[str, Any]:
    """
    Seed common RNG sources and return structured metadata.

    Notes:
    - Setting PYTHONHASHSEED is only effective at process start. We record an
      intent value but do not attempt to mutate it here.
    - Torch deterministic settings can reduce performance and may not cover all
      ops. This is best-effort.
    """
    seed_int = int(seed)

    # Record (but do not force) PYTHONHASHSEED. If unset, we suggest one via metadata.
    pyhash_env = os.environ.get("PYTHONHASHSEED")
    pyhash_val: Optional[int]
    try:
        pyhash_val = int(pyhash_env) if pyhash_env is not None and pyhash_env != "" else None
    except Exception:
        pyhash_val = None

    random.seed(seed_int)
    random_seeded = True

    numpy_seeded = False
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed_int)
        numpy_seeded = True
    except Exception:
        numpy_seeded = False

    torch_seeded = False
    torch_det: Optional[bool] = None
    cudnn_det: Optional[bool] = None
    cudnn_bench: Optional[bool] = None
    try:
        import torch  # type: ignore

        torch.manual_seed(seed_int)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_int)
        torch_seeded = True

        # Determinism flags (best-effort)
        if deterministic_torch:
            try:
                torch.use_deterministic_algorithms(True)
                torch_det = True
            except Exception:
                torch_det = None
            try:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                cudnn_det = True
            except Exception:
                cudnn_det = None
            try:
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
                cudnn_bench = False
            except Exception:
                cudnn_bench = None
        else:
            torch_det = False
    except Exception:
        torch_seeded = False

    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
    }
    for name in ("numpy", "torch"):
        v = _safe_version(name)
        if v:
            versions[name] = v

    info = SeedInfo(
        seed=seed_int,
        python_hash_seed=pyhash_val,
        random_seeded=random_seeded,
        numpy_seeded=numpy_seeded,
        torch_seeded=torch_seeded,
        torch_deterministic=torch_det,
        torch_cudnn_deterministic=cudnn_det,
        torch_cudnn_benchmark=cudnn_bench,
        versions=versions,
    )
    return info.as_dict()

