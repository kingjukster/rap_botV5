#!/usr/bin/env python3
"""
AI-assisted labeling for metric benchmark verses (optional OpenAI).

Requires: pip install openai and OPENAI_API_KEY in the environment.

Reads unlabeled JSONL, writes labeled JSONL with labels + label_provenance=ai.

By default, API calls follow harvest.metric_benchmark_stratum order: technical → semantic →
modern → control → backfill, then verses with missing/unknown stratum. Use --stop-after-stratum
to only label through a given tier; output stays in original JSONL line order.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo_rhyme.metric_benchmark.schema import LABEL_KEYS
from evo_rhyme.metric_benchmark.protocol import default_protocol_manifest, load_protocol_manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Baseline filter strata (tier 1→5); unknown / missing harvest stratum sorts last.
STRATUM_LABEL_ORDER: tuple[str, ...] = ("technical", "semantic", "modern", "control", "backfill")


def row_stratum_rank(row: Dict[str, Any]) -> int:
    """Sort key: lower = earlier in labeling pipeline (technical first)."""
    h = row.get("harvest")
    if not isinstance(h, dict):
        return len(STRATUM_LABEL_ORDER)
    raw = (h.get("metric_benchmark_stratum") or "").strip().lower()
    if raw in STRATUM_LABEL_ORDER:
        return STRATUM_LABEL_ORDER.index(raw)
    return len(STRATUM_LABEL_ORDER)


def max_stratum_rank_inclusive(stop_after: Optional[str]) -> int:
    """Highest stratum index (inclusive) to send to the API; None = all strata + unknown."""
    if not stop_after:
        return len(STRATUM_LABEL_ORDER)  # allow unknown bucket too
    s = stop_after.strip().lower()
    if s not in STRATUM_LABEL_ORDER:
        raise SystemExit(
            f"--stop-after-stratum must be one of {list(STRATUM_LABEL_ORDER)} (got {stop_after!r})"
        )
    return STRATUM_LABEL_ORDER.index(s)


def build_pending_label_pairs(
    merged: List[Dict[str, Any]],
    *,
    stop_after: Optional[str],
    force: bool,
    label_order: str,
) -> List[tuple[int, int]]:
    """
    Pairs (sort_key, index) for rows that still need labeling, respecting stop-after-stratum.

    label_order 'stratum': sort by (stratum_rank, index).
    label_order 'input':   original JSONL line order.
    """
    cap = max_stratum_rank_inclusive(stop_after)
    idxs: List[int] = []
    for i, cur in enumerate(merged):
        if not force and _has_full_labels(cur):
            continue
        r = row_stratum_rank(cur)
        if r > cap:
            continue
        idxs.append(i)
    if label_order == "input":
        return [(0, i) for i in idxs]
    pairs = [(row_stratum_rank(merged[i]), i) for i in idxs]
    pairs.sort(key=lambda t: (t[0], t[1]))
    return pairs


def _write_output_rows(
    path: Path,
    merged: List[Dict[str, Any]],
    labeled_by_idx: Dict[int, Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out_f:
        for i in range(len(merged)):
            row = labeled_by_idx.get(i, merged[i])
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()


def _build_user_prompt(verse_text: str, manifest: Dict[str, Any]) -> str:
    flow = manifest.get("flow_definition", {})
    punch = manifest.get("punchline_definition_v1", "")
    flu = manifest.get("fluency_definition_v1", "")
    ori = manifest.get("originality_definition_v1", "")
    anchors = "\n".join(f"  {k}: {v}" for k, v in sorted(flow.items(), key=lambda x: -float(x[0])))
    keys = ", ".join(f'"{k}"' for k in LABEL_KEYS)
    return f"""Rate this rap verse. Output ONLY a single JSON object with these keys, each a float 0.0-1.0:
{keys}

Flow anchors (interpolate between):
{anchors}

Punchline: {punch}
Fluency: {flu}
Originality: {ori}

Verse (one bar per line):
{verse_text}
"""


def is_insufficient_quota_error(err: BaseException) -> bool:
    """True when OpenAI returns billing/plan exhaustion (do not retry)."""
    if "insufficient_quota" in str(err).lower():
        return True
    body = getattr(err, "body", None)
    if isinstance(body, dict) and (body.get("error") or {}).get("code") == "insufficient_quota":
        return True
    resp = getattr(err, "response", None)
    if resp is not None:
        try:
            j = resp.json()
            if (j.get("error") or {}).get("code") == "insufficient_quota":
                return True
        except Exception:
            pass
    return False


def _call_openai_once(client: Any, prompt: str, model: str) -> Dict[str, float]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict rap lyrics evaluator. Reply with JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) >= 2 else text
        if text.startswith("json"):
            text = text[4:].lstrip()
    data = json.loads(text)
    out: Dict[str, float] = {}
    for k in LABEL_KEYS:
        out[k] = max(0.0, min(1.0, float(data[k])))
    return out


def _call_openai(
    prompt: str,
    model: str,
    *,
    max_retries: int,
    base_delay_s: float,
) -> Dict[str, float]:
    try:
        from openai import APIConnectionError, OpenAI, RateLimitError
    except ImportError as e:
        raise RuntimeError("Install openai package: pip install openai") from e
    client = OpenAI()
    last_err: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            return _call_openai_once(client, prompt, model)
        except RateLimitError as e:
            last_err = e
            if is_insufficient_quota_error(e):
                raise
            delay = base_delay_s * (2**attempt) + random.uniform(0.0, 0.5)
            logger.warning("Rate limited (attempt %d/%d), sleeping %.1fs", attempt + 1, max_retries, delay)
            time.sleep(delay)
        except APIConnectionError as e:
            last_err = e
            delay = base_delay_s * (2**attempt) + random.uniform(0.0, 0.5)
            logger.warning("Connection error (attempt %d/%d): %s; sleeping %.1fs", attempt + 1, max_retries, e, delay)
            time.sleep(delay)
    assert last_err is not None
    raise last_err


def _load_jsonl_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row.get("id")
            if isinstance(rid, str):
                out[rid] = row
    return out


def _has_full_labels(row: Dict[str, Any]) -> bool:
    lab = row.get("labels")
    if not isinstance(lab, dict):
        return False
    return all(k in lab for k in LABEL_KEYS)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=ROOT / "data/metric_benchmark/protocol_defaults.json")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--limit", type=int, default=0, help="Max verses to send to API (0 = all unlabeled)")
    p.add_argument("--force", action="store_true", help="Re-label even if labels exist")
    p.add_argument("--dry-run", action="store_true", help="Print first prompt only; no API")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse labeled rows from existing --output by verse id (same --input order); use after crash or quota stop",
    )
    p.add_argument("--max-retries", type=int, default=10, dest="max_retries", help="Retries for transient rate/connection errors")
    p.add_argument(
        "--retry-base-delay",
        type=float,
        default=1.0,
        dest="retry_base_delay",
        help="Initial backoff (seconds) for 429 / connection errors (exponential)",
    )
    p.add_argument(
        "--sleep-after-request",
        type=float,
        default=0.15,
        dest="sleep_after_request",
        help="Pause after each successful API call to reduce rate limits (0 to disable)",
    )
    p.add_argument(
        "--stop-after-stratum",
        type=str,
        default=None,
        metavar="STRATUM",
        help=(
            "Only call the API for verses in harvest.metric_benchmark_stratum up through this tier "
            f"(order: {' → '.join(STRATUM_LABEL_ORDER)}). Omit to label all strata; unknown/missing stratum is last."
        ),
    )
    p.add_argument(
        "--label-order",
        type=str,
        choices=("stratum", "input"),
        default="stratum",
        help="stratum: technical→…→backfill then unknown; input: original JSONL line order",
    )
    args = p.parse_args()

    manifest = load_protocol_manifest(args.protocol) if args.protocol.exists() else default_protocol_manifest()

    rows: List[Dict[str, Any]] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        existing_dr = _load_jsonl_by_id(args.output) if args.resume else {}
        merged_preview: List[Dict[str, Any]] = []
        for row in rows:
            current = dict(row)
            rid = current.get("id")
            if (
                args.resume
                and isinstance(rid, str)
                and rid in existing_dr
                and _has_full_labels(existing_dr[rid])
                and not args.force
            ):
                current = dict(existing_dr[rid])
            merged_preview.append(current)
        pairs = build_pending_label_pairs(
            merged_preview,
            stop_after=args.stop_after_stratum,
            force=args.force,
            label_order=args.label_order,
        )
        sample_idx = pairs[0][1] if pairs else 0
        sample = merged_preview[sample_idx] if merged_preview else rows[0]
        verse_text = "\n".join(str(x) for x in sample.get("lyrics", []))
        logger.info(
            "Dry-run: %d verses queued for API in this run (first verse stratum=%s, order=%s)",
            len(pairs),
            (sample.get("harvest") or {}).get("metric_benchmark_stratum", "?"),
            args.label_order,
        )
        logger.info("Dry-run prompt sample:\n%s", _build_user_prompt(verse_text, manifest)[:2000])
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set.")
        return 1

    existing_by_id = _load_jsonl_by_id(args.output) if args.resume else {}

    merged: List[Dict[str, Any]] = []
    for row in rows:
        current = dict(row)
        rid = current.get("id")
        if (
            args.resume
            and isinstance(rid, str)
            and rid in existing_by_id
            and _has_full_labels(existing_by_id[rid])
            and not args.force
        ):
            current = dict(existing_by_id[rid])
        merged.append(current)

    pending_pairs = build_pending_label_pairs(
        merged,
        stop_after=args.stop_after_stratum,
        force=args.force,
        label_order=args.label_order,
    )
    if args.stop_after_stratum:
        logger.info(
            "Labeling order=%s, stop-after-stratum=%s (%d verses queued for API, %d total lines)",
            args.label_order,
            args.stop_after_stratum,
            len(pending_pairs),
            len(rows),
        )
    else:
        logger.info(
            "Labeling order=%s through all strata (%d verses queued for API, %d total lines)",
            args.label_order,
            len(pending_pairs),
            len(rows),
        )

    labeled_by_idx: Dict[int, Dict[str, Any]] = {}
    api_count = 0

    try:
        for _rank, i in pending_pairs:
            if args.limit > 0 and api_count >= args.limit:
                break
            cur = merged[i]
            lyrics = cur.get("lyrics") or []
            verse_text = "\n".join(str(x) for x in lyrics)
            prompt = _build_user_prompt(verse_text, manifest)
            try:
                labels = _call_openai(
                    prompt,
                    args.model,
                    max_retries=args.max_retries,
                    base_delay_s=args.retry_base_delay,
                )
            except Exception as e:
                if is_insufficient_quota_error(e):
                    logger.error(
                        "OpenAI insufficient_quota: add billing/credits at "
                        "https://platform.openai.com/account/billing — "
                        "writing partial output; rerun with --resume after fixing billing."
                    )
                    _write_output_rows(args.output, merged, labeled_by_idx)
                    return 1
                raise
            prov = {k: "ai" for k in LABEL_KEYS}
            new_row = {**cur, "labels": labels, "label_provenance": prov, "unlabeled": False}
            new_row.pop("split", None)
            labeled_by_idx[i] = new_row
            api_count += 1
            if args.sleep_after_request > 0:
                time.sleep(args.sleep_after_request)
            if api_count % 10 == 0:
                logger.info("Labeled %d verses via API", api_count)
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted during API pass; writing partial output; continue with --resume (%s)",
            args.output,
        )
        _write_output_rows(args.output, merged, labeled_by_idx)
        return 130

    _write_output_rows(args.output, merged, labeled_by_idx)
    logger.info("API calls: %d; wrote %s", api_count, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
