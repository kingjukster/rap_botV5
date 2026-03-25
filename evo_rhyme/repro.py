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
from pathlib import Path
from typing import Any, Dict, Optional

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone


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


def repo_root() -> Path:
    """Project root (parent of the `evo_rhyme` package)."""
    return Path(__file__).resolve().parents[1]


def get_git_commit_sha(cwd: Optional[Path] = None) -> Optional[str]:
    """Best-effort `git rev-parse HEAD` for reproducibility metadata."""
    root = cwd or repo_root()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def sha256_file(path: Path) -> Optional[str]:
    """SHA256 hex digest of file contents, or None if missing/unreadable."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def learned_policy_sha256(root: Optional[Path] = None) -> Optional[str]:
    """Hash of `artifacts/learned_policy.json` when present."""
    r = root or repo_root()
    p = r / "artifacts" / "learned_policy.json"
    return sha256_file(p) if p.is_file() else None


def run_provenance_dict(
    *,
    extra: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Structured provenance for run artifacts (config.json, JSONL, etc.)."""
    root = project_root or repo_root()
    prov: Dict[str, Any] = {
        "git_commit_sha": get_git_commit_sha(root),
        "learned_policy_sha256": learned_policy_sha256(root),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    env_policy = os.environ.get("RAPBOT_POLICY_SHA256", "").strip()
    if env_policy:
        prov["policy_sha256_env"] = env_policy
    if extra:
        prov.update(extra)
    return prov


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON object per line (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)

