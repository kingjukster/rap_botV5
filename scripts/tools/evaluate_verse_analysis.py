#!/usr/bin/env python
"""
Evaluation framework for the Two-Track Verse Analysis System.

Loads labeled dataset (JSON/JSONL), runs verse analysis, and computes:
- Rhyme span precision/recall/F1 vs gold spans
- Family clustering purity + Adjusted Rand Index
- Chain accuracy F1 vs gold chains
- Metaphor frame F1
- Punchline F1
- Cohen's kappa for multi-annotator agreement
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rapbot.verse_analyzer import analyze_verse
from rapbot.verse_annotation import VerseAnnotation


# ---------------------------------------------------------------------------
# Labeled data format (per item in JSONL or single JSON)
# ---------------------------------------------------------------------------
# {
#   "verse": "raw verse text",
#   "lines": ["line1", "line2", ...],  # optional, derived from verse if absent
#   "labels": {
#     "rhyme_spans": [[line_idx, word_idx, word_idx], ...],  # (line, start_word, end_word) per span
#     "rhyme_families": [[span_idx1, span_idx2], ...],  # gold clusters (indices into rhyme_spans)
#     "chains": [[[line_idx, end_word], ...], ...],  # gold chains: list of (line_idx, end_word) per chain
#     "metaphor_frames": [{"source_domain": "x", "target_domain": "y", "spans": [[line_start, line_end], ...]}],
#     "punchlines": [{"line_index": 0, "pivot_word": "..."}],
#   },
#   "annotators": {"ann1": {...}, "ann2": {...}},  # optional: multiple annotators for kappa
# }


def _load_labeled(path: Path) -> List[Dict[str, Any]]:
    """Load labeled data from JSON or JSONL."""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        items = []
        for line in text.strip().split("\n"):
            if line.strip():
                items.append(json.loads(line))
        return items


def _predicted_rhyme_spans(ann: VerseAnnotation, lines: List[str]) -> Set[Tuple[int, int]]:
    """Extract predicted rhyme (line_idx, word_idx) from clusters + chain end words."""
    spans = set()
    for cluster in ann.rhyme_clusters:
        for s in cluster.spans:
            for wi in s.word_indices:
                spans.add((s.line_idx, wi))
    for chain in ann.rhyme_chains:
        for line_idx, end_word in chain.positions:
            if line_idx < len(lines):
                words = lines[line_idx].split()
                for wi, w in enumerate(words):
                    if w.rstrip(".,!?;:'\"").lower() == end_word.lower():
                        spans.add((line_idx, wi))
                        break
    return spans


def _gold_rhyme_spans(labels: Dict) -> Set[Tuple[int, int]]:
    """Extract gold rhyme (line_idx, word_idx) from labels.rhyme_spans."""
    spans = set()
    for item in labels.get("rhyme_spans", []):
        if len(item) >= 3:
            for wi in range(item[1], item[2] + 1):
                spans.add((item[0], wi))
        elif len(item) >= 2:
            spans.add((item[0], item[1]))
    return spans


def _compute_span_f1(
    pred: Set[Tuple[int, ...]],
    gold: Set[Tuple[int, ...]],
) -> Dict[str, float]:
    """Precision, recall, F1 for span overlap (exact match)."""
    if not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0} if not pred else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(pred & gold)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def _adjusted_rand_index(
    pred_clusters: List[Set[Tuple[int, int]]],
    gold_clusters: List[Set[Tuple[int, int]]],
) -> float:
    """
    Adjusted Rand Index for clustering. Uses intersection of pred and gold positions.
    Returns ARI in [-1, 1]; 1 = perfect agreement.
    """
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return 0.0
    all_positions = set()
    for s in pred_clusters:
        all_positions.update(s)
    for s in gold_clusters:
        all_positions.update(s)
    if not all_positions:
        return 1.0
    pos_list = sorted(all_positions)
    pos_to_idx = {p: i for i, p in enumerate(pos_list)}
    gold_labels = [-1] * len(pos_list)
    pred_labels = [-1] * len(pos_list)
    for gid, gs in enumerate(gold_clusters):
        for p in gs:
            if p in pos_to_idx:
                gold_labels[pos_to_idx[p]] = gid
    for pid, ps in enumerate(pred_clusters):
        for p in ps:
            if p in pos_to_idx:
                pred_labels[pos_to_idx[p]] = pid
    # Use positions that appear in both gold and pred
    valid = [i for i in range(len(pos_list)) if gold_labels[i] >= 0 and pred_labels[i] >= 0]
    if not valid:
        has_gold = any(gold_labels[i] >= 0 for i in range(len(pos_list)))
        return 1.0 if not has_gold else 0.0
    return float(adjusted_rand_score(
        [gold_labels[i] for i in valid],
        [pred_labels[i] for i in valid],
    ))


def _compute_clustering_purity(pred_clusters: List[Set], gold_clusters: List[Set]) -> float:
    """
    Purity: for each predicted cluster, what fraction of items belong to the
    majority gold cluster? Average over predicted clusters.
    """
    if not pred_clusters:
        return 0.0
    total = 0.0
    for pset in pred_clusters:
        if not pset:
            continue
        best_overlap = 0
        for gset in gold_clusters:
            overlap = len(pset & gset)
            if overlap > best_overlap:
                best_overlap = overlap
        total += best_overlap / len(pset) if pset else 0
    return total / len(pred_clusters)


def _predicted_metaphor_spans(ann: VerseAnnotation) -> Set[Tuple[int, int]]:
    """(line_start, line_end) for metaphor frames."""
    spans = set()
    for m in ann.metaphor_frames:
        for s in m.spans:
            spans.add((s[0], s[1]))
    return spans


def _gold_metaphor_spans(labels: Dict) -> Set[Tuple[int, int]]:
    spans = set()
    for m in labels.get("metaphor_frames", []):
        for s in m.get("spans", []):
            if len(s) >= 2:
                spans.add((s[0], s[1]))
    return spans


def _predicted_chain_positions(ann: VerseAnnotation) -> Set[Tuple[int, str]]:
    """(line_idx, end_word) from predicted rhyme chains."""
    positions = set()
    for chain in ann.rhyme_chains:
        for line_idx, word in chain.positions:
            positions.add((line_idx, word.lower()))
    return positions


def _gold_chain_positions(labels: Dict) -> Set[Tuple[int, str]]:
    """(line_idx, end_word) from labels.chains."""
    positions = set()
    for chain in labels.get("chains", []):
        for item in chain:
            if len(item) >= 2:
                positions.add((item[0], str(item[1]).lower()))
            elif len(item) == 1:
                positions.add((item[0], ""))
    return positions


def _predicted_punchlines(ann: VerseAnnotation) -> Set[Tuple[int, str]]:
    """(line_index, pivot_word) from predicted punchlines."""
    return {(p.line_index, p.pivot_word.lower()) for p in ann.punchlines}


def _gold_punchlines(labels: Dict) -> Set[Tuple[int, str]]:
    """(line_index, pivot_word) from labels.punchlines."""
    positions = set()
    for p in labels.get("punchlines", []):
        li = p.get("line_index", 0)
        pw = p.get("pivot_word", "")
        positions.add((li, str(pw).lower()))
    return positions


def _cohens_kappa(ann1_items: List[Any], ann2_items: List[Any]) -> float:
    """
    Cohen's kappa for binary/categorical agreement.
    Items should be aligned (same verse, same span index).
    """
    if len(ann1_items) != len(ann2_items) or not ann1_items:
        return 0.0
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(ann1_items, ann2_items))
    except ImportError:
        # Manual computation
        n = len(ann1_items)
        categories = set(ann1_items) | set(ann2_items)
        p_o = sum(1 for i in range(n) if ann1_items[i] == ann2_items[i]) / n
        p_e = 0.0
        for c in categories:
            p1 = sum(1 for x in ann1_items if x == c) / n
            p2 = sum(1 for x in ann2_items if x == c) / n
            p_e += p1 * p2
        if p_e >= 1.0:
            return 0.0
        return (p_o - p_e) / (1 - p_e)


def evaluate_item(
    item: Dict[str, Any],
    run_analysis: bool = True,
) -> Dict[str, Any]:
    """Evaluate single labeled item."""
    verse = item.get("verse", "")
    lines = item.get("lines") or [ln.strip() for ln in verse.strip().split("\n") if ln.strip()]
    labels = item.get("labels", {})

    if run_analysis:
        ann = analyze_verse(verse, use_embeddings=False)  # Faster for eval
    else:
        ann = None

    result = {"verse_preview": verse[:80] + "..." if len(verse) > 80 else verse}

    if not labels:
        return result

    # Rhyme span F1 (clusters + chains)
    gold_rhyme = _gold_rhyme_spans(labels)
    if gold_rhyme and ann:
        pred_rhyme = _predicted_rhyme_spans(ann, lines)
        result["rhyme_span_f1"] = _compute_span_f1(pred_rhyme, gold_rhyme)

    # Chain F1 (end-word positions)
    gold_chains = _gold_chain_positions(labels)
    if gold_chains and ann:
        pred_chains = _predicted_chain_positions(ann)
        result["chain_f1"] = _compute_span_f1(pred_chains, gold_chains)

    # Metaphor frame F1 (span-level)
    gold_meta = _gold_metaphor_spans(labels)
    if gold_meta and ann:
        pred_meta = _predicted_metaphor_spans(ann)
        result["metaphor_frame_f1"] = _compute_span_f1(pred_meta, gold_meta)

    # Punchline F1
    gold_punch = _gold_punchlines(labels)
    if gold_punch and ann:
        pred_punch = _predicted_punchlines(ann)
        result["punchline_f1"] = _compute_span_f1(pred_punch, gold_punch)

    # Family clustering purity + ARI (if gold families provided)
    gold_families = labels.get("rhyme_families", [])
    rhyme_span_list = labels.get("rhyme_spans", [])
    if gold_families and rhyme_span_list and ann:
        gold_span_list = [(it[0], it[1], it[2] if len(it) > 2 else it[1]) for it in rhyme_span_list]
        pred_sets_impl = []
        for c in ann.rhyme_clusters:
            s = set()
            for sp in c.spans:
                for wi in sp.word_indices:
                    s.add((sp.line_idx, wi))
            if s:
                pred_sets_impl.append(s)
        gold_sets_impl = []
        for fam in gold_families:
            gs = set()
            for idx in fam:
                if idx < len(gold_span_list):
                    entry = gold_span_list[idx]
                    line_idx, w_start = entry[0], entry[1]
                    w_end = entry[2] if len(entry) > 2 else w_start
                    for wi in range(w_start, w_end + 1):
                        gs.add((line_idx, wi))
            if gs:
                gold_sets_impl.append(gs)
        if pred_sets_impl and gold_sets_impl:
            result["family_clustering_purity"] = _compute_clustering_purity(pred_sets_impl, gold_sets_impl)
            result["family_clustering_ari"] = _adjusted_rand_index(pred_sets_impl, gold_sets_impl)

    # Cohen's kappa (if multiple annotators)
    annotators = item.get("annotators", {})
    if len(annotators) >= 2:
        keys = list(annotators.keys())
        a1 = annotators[keys[0]]
        a2 = annotators[keys[1]]
        # Flatten to comparable lists (e.g. rhyme present/absent per span)
        # Simplified: binary per-line "has_punchline" or similar
        p1_labels = a1.get("punchline_lines", [])
        p2_labels = a2.get("punchline_lines", [])
        if p1_labels and p2_labels and len(p1_labels) == len(p2_labels):
            result["cohens_kappa"] = _cohens_kappa(p1_labels, p2_labels)

    return result


def aggregate_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate per-item metrics into summary."""
    agg = {}
    rhyme_f1s = [r["rhyme_span_f1"] for r in results if "rhyme_span_f1" in r]
    if rhyme_f1s:
        agg["rhyme_span"] = {
            "precision_mean": sum(r["precision"] for r in rhyme_f1s) / len(rhyme_f1s),
            "recall_mean": sum(r["recall"] for r in rhyme_f1s) / len(rhyme_f1s),
            "f1_mean": sum(r["f1"] for r in rhyme_f1s) / len(rhyme_f1s),
            "n": len(rhyme_f1s),
        }
    chain_f1s = [r["chain_f1"] for r in results if "chain_f1" in r]
    if chain_f1s:
        agg["chain"] = {
            "precision_mean": sum(r["precision"] for r in chain_f1s) / len(chain_f1s),
            "recall_mean": sum(r["recall"] for r in chain_f1s) / len(chain_f1s),
            "f1_mean": sum(r["f1"] for r in chain_f1s) / len(chain_f1s),
            "n": len(chain_f1s),
        }
    meta_f1s = [r["metaphor_frame_f1"] for r in results if "metaphor_frame_f1" in r]
    if meta_f1s:
        agg["metaphor_frame"] = {
            "precision_mean": sum(r["precision"] for r in meta_f1s) / len(meta_f1s),
            "recall_mean": sum(r["recall"] for r in meta_f1s) / len(meta_f1s),
            "f1_mean": sum(r["f1"] for r in meta_f1s) / len(meta_f1s),
            "n": len(meta_f1s),
        }
    punch_f1s = [r["punchline_f1"] for r in results if "punchline_f1" in r]
    if punch_f1s:
        agg["punchline"] = {
            "precision_mean": sum(r["precision"] for r in punch_f1s) / len(punch_f1s),
            "recall_mean": sum(r["recall"] for r in punch_f1s) / len(punch_f1s),
            "f1_mean": sum(r["f1"] for r in punch_f1s) / len(punch_f1s),
            "n": len(punch_f1s),
        }
    purities = [r["family_clustering_purity"] for r in results if "family_clustering_purity" in r]
    if purities:
        agg["family_clustering_purity"] = {"mean": sum(purities) / len(purities), "n": len(purities)}
    aris = [r["family_clustering_ari"] for r in results if "family_clustering_ari" in r]
    if aris:
        agg["family_clustering_ari"] = {"mean": sum(aris) / len(aris), "n": len(aris)}
    kappas = [r["cohens_kappa"] for r in results if "cohens_kappa" in r]
    if kappas:
        agg["cohens_kappa"] = {"mean": sum(kappas) / len(kappas), "n": len(kappas)}
    return agg


def main():
    parser = argparse.ArgumentParser(description="Evaluate verse analysis on labeled data")
    parser.add_argument("--labeled_path", "-l", type=Path, required=True, help="JSON or JSONL of labeled verses")
    parser.add_argument("--output_metrics", "-o", type=Path, help="Write metrics JSON here")
    parser.add_argument("--no-run", action="store_true", help="Skip running analysis (only for kappa from labels)")
    args = parser.parse_args()

    items = _load_labeled(args.labeled_path)
    if not items:
        print("No labeled items found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(items)} labeled item(s)")
    results = []
    for i, item in enumerate(items):
        r = evaluate_item(item, run_analysis=not args.no_run)
        results.append(r)

    agg = aggregate_metrics(results)
    print("\n=== Aggregated Metrics ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    if args.output_metrics:
        args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
        out = {"per_item": results, "aggregate": agg}
        args.output_metrics.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nMetrics saved to: {args.output_metrics}")


if __name__ == "__main__":
    main()
