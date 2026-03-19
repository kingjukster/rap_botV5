#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo_rhyme import db as rapdb  # noqa: E402


def iter_target_files(root: Path, *, include_runs: bool, include_results: bool) -> Iterable[Path]:
    if include_runs:
        runs_dir = root / "data" / "evo_rhyme" / "runs"
        if runs_dir.exists():
            yield from runs_dir.rglob("*.json")
    if include_results:
        yield from root.glob("results_qd*.json")


def iter_explicit_files(root: Path, paths: Sequence[str]) -> Iterable[Path]:
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = (root / p).resolve()
        yield p


def rel_path(root: Path, p: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return p.as_posix()


def infer_run_tag(rel: str) -> Optional[str]:
    parts = rel.split("/")
    if len(parts) >= 4 and parts[0] == "data" and parts[1] == "evo_rhyme" and parts[2] == "runs":
        return parts[3] or None
    return None


def infer_kind(p: Path) -> str:
    name = p.name.lower()
    if name == "archive.json":
        return "run_archive"
    if name == "config.json":
        return "run_config"
    if name == "top_candidates.json":
        return "run_top_candidates"
    if name.startswith("results_qd") and name.endswith(".json"):
        return "results_qd"
    if name.startswith("results_") and name.endswith(".json"):
        return "results"
    if name.endswith("_weights_quick.json") or name.endswith("_weights_default.json") or name.endswith("weights_evolved.json"):
        return "weights"
    return "json"


def validate_json(content: bytes, *, display_name: str) -> None:
    try:
        json.loads(content.decode("utf-8"))
    except UnicodeDecodeError as e:
        raise ValueError(f"{display_name}: not utf-8 ({e})") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"{display_name}: invalid json ({e})") from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Archive run/result JSON artifacts into MySQL (gzipped), optionally deleting originals."
    )
    parser.add_argument("--root", default=".", help="Repo root (default: .)")
    parser.add_argument("--include-runs", action="store_true", help="Include data/evo_rhyme/runs/**/*.json")
    parser.add_argument("--include-results", action="store_true", help="Include results_qd*.json at repo root")
    parser.add_argument("files", nargs="*", help="Explicit file paths to archive (relative to repo root)")
    parser.add_argument("--delete", action="store_true", help="Delete files after successful insert")
    parser.add_argument("--dry-run", action="store_true", help="Do not insert or delete; just report actions")
    parser.add_argument("--limit", type=int, default=0, help="Max files processed (0 = no limit)")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not (args.include_runs or args.include_results or args.files):
        print("Nothing selected: use --include-runs and/or --include-results and/or pass file paths")
        return 2

    if not rapdb.db_enabled():
        print("DB disabled. Set RAPBOT_USE_DB=1 and RAPBOT_DB_* env vars.")
        return 2

    file_set = set(
        iter_target_files(root, include_runs=args.include_runs, include_results=args.include_results)
    ) | set(iter_explicit_files(root, args.files))
    files = sorted(file_set, key=lambda p: str(p))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    ok = 0
    skipped = 0
    failed = 0

    for p in files:
        rp = rel_path(root, p)
        try:
            if not p.exists() or not p.is_file():
                skipped += 1
                print(f"SKIP missing {rp}")
                continue
            content = p.read_bytes()
            validate_json(content, display_name=rp)
            sha = hashlib.sha256(content).hexdigest()
            kind = infer_kind(p)
            run_tag = infer_run_tag(rp)
            if args.dry_run:
                print(f"DRY-RUN insert kind={kind} sha256={sha} path={rp}")
            else:
                aid = rapdb.insert_artifact(kind=kind, content_bytes=content, sha256=sha, run_tag=run_tag, rel_path=rp)
                if aid == -1:
                    raise RuntimeError("DB insert failed")
                print(f"Inserted artifact_id={aid} kind={kind} sha256={sha} path={rp}")

            if args.delete:
                if args.dry_run:
                    print(f"DRY-RUN delete {rp}")
                else:
                    p.unlink(missing_ok=True)
                    print(f"Deleted {rp}")
            ok += 1
        except Exception as e:
            failed += 1
            print(f"FAILED {rp}: {e}")
            continue

    print(f"Done. ok={ok} failed={failed} skipped={skipped} total={len(files)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

