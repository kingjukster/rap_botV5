#!/usr/bin/env python
"""
Generate a dashboard HTML from an existing verse analysis JSON file.
No re-analysis needed - just loads JSON and renders the dashboard.

Usage:
    python scripts/tools/view_dashboard.py
    python scripts/tools/view_dashboard.py data/verse_analysis_output.json
    python scripts/tools/view_dashboard.py data/verse_analysis_output.json -o data/dashboard.html
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rapbot.verse_annotation import VerseAnnotation
from rapbot.verse_analyzer import analyze_verse
from scripts.tools.analyze_verse import format_html, format_html_v4, format_html_v5, DEFAULT_VERSE


def dict_to_annotation(d: dict) -> VerseAnnotation:
    """Reconstruct VerseAnnotation from JSON dict (from to_dict())."""
    from rapbot.verse_annotation import Token
    from rapbot.internal_rhyme_detector import RhymeCluster, Span
    from rapbot.rhyme_chain_detector import RhymeChain
    from rapbot.discourse_analyzer import DiscourseUnit
    from rapbot.metaphor_detector import MetaphorFrame
    from rapbot.punchline_detector import Punchline

    def span_from_dict(s):
        return Span(
            text=s["text"],
            start_char=s["start_char"],
            end_char=s["end_char"],
            line_idx=s["line_idx"],
            word_indices=tuple(s.get("word_indices", [])),
        )

    def token_from_dict(t):
        return Token(
            word=t["word"],
            line_idx=t["line_idx"],
            word_idx=t["word_idx"],
            char_start=t["char_start"],
            char_end=t["char_end"],
            token_id=t.get("token_id"),
            anchor_status=t.get("anchor_status"),
            source=t.get("source"),
            phonemes=t.get("phonemes"),
            tail=t.get("tail"),
            nucleus=t.get("nucleus"),
        )

    tokens = [token_from_dict(t) for t in d.get("tokens", [])]

    rhyme_clusters = []
    for c in d.get("rhyme_clusters", []):
        cluster = RhymeCluster(
            family_id=c["family_id"],
            spans=[span_from_dict(s) for s in c["spans"]],
            confidence=c.get("confidence", 0.0),
            rhyme_type=c.get("rhyme_type") or "rhyme",
        )
        if c.get("representative_tail") is not None:
            cluster.representative_tail = tuple(c["representative_tail"])
        if c.get("top_anchors") is not None:
            cluster.top_anchors = c["top_anchors"]
        if c.get("example_spans") is not None:
            cluster.example_spans = c["example_spans"]
        rhyme_clusters.append(cluster)

    rhyme_chains = []
    for i, c in enumerate(d.get("rhyme_chains", [])):
        chain = RhymeChain(
            chain_id=c.get("chain_id", i),
            pattern=c.get("pattern", ""),
            positions=[(p[0], p[1]) for p in c.get("positions", [])],
            family_id=c.get("family_id"),
            confidence=c.get("confidence", 0.0),
            chain_continuity_score=c.get("chain_continuity_score", 0.0),
            chain_type=c.get("chain_type", "tail"),
            representative_nucleus=c.get("representative_nucleus"),
            representative_tail=c.get("representative_tail"),
            positions_with_phonemes=c.get("positions_with_phonemes"),
        )
        rhyme_chains.append(chain)

    metaphor_frames = [
        MetaphorFrame(
            source_domain=m["source_domain"],
            target_domain=m["target_domain"],
            spans=[(s[0], s[1]) for s in m.get("spans", [])],
            confidence=m.get("confidence", 0.0),
            cue_words=m.get("cue_words", []),
        )
        for m in d.get("metaphor_frames", [])
    ]

    punchlines = [
        Punchline(
            pivot_word=p["pivot_word"],
            line=p.get("line", ""),
            line_index=p.get("line_index", 0),
            original_meaning=p.get("original_meaning", ""),
            double_meaning=p.get("double_meaning", ""),
            confidence=p.get("confidence", 0.0),
        )
        for p in d.get("punchlines", [])
    ]

    discourse_units = [
        DiscourseUnit(
            type=d["type"],
            span=(d["span"][0], d["span"][1]),
            confidence=d.get("confidence", 0.0),
        )
        for d in d.get("discourse_units", [])
    ]

    stance_labels = {}
    for k, v in d.get("stance_labels", {}).items():
        try:
            stance_labels[int(k)] = v
        except ValueError:
            stance_labels[k] = v

    return VerseAnnotation(
        verse=d.get("verse", ""),
        lines=d.get("lines", []),
        tokens=tokens,
        rhyme_clusters=rhyme_clusters,
        rhyme_chains=rhyme_chains,
        metaphor_frames=metaphor_frames,
        punchlines=punchlines,
        discourse_units=discourse_units,
        stance_labels=stance_labels,
        cohesion_metrics=d.get("cohesion_metrics", {}),
        density_metrics=d.get("density_metrics", {}),
        confidence_scores=d.get("confidence_scores", {}),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate dashboard HTML from verse analysis JSON"
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        type=Path,
        default=Path("data/verse_analysis_output.json"),
        help="Input JSON file (default: data/verse_analysis_output.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML file (default: same dir as input, name ends with .html)",
    )
    parser.add_argument(
        "--v4",
        action="store_true",
        help="Use v4 dashboard format (Anchor view, anchor_status legend)",
    )
    parser.add_argument(
        "--v5",
        action="store_true",
        help="Use v5 dashboard format (Chain Inspector, phoneme tooltips)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Re-analyze verse instead of loading JSON (uses default verse)",
    )
    args = parser.parse_args()

    if args.analyze:
        print("Analyzing verse...")
        ann = analyze_verse(DEFAULT_VERSE, use_embeddings=False)
    elif args.json_file.exists():
        print(f"Loading {args.json_file}...")
        d = json.loads(args.json_file.read_text(encoding="utf-8"))
        ann = dict_to_annotation(d)
    else:
        print(f"Error: {args.json_file} not found.")
        sys.exit(1)

    if args.v5:
        html = format_html_v5(ann)
    elif args.v4:
        html = format_html_v4(ann)
    else:
        html = format_html(ann)

    out_path = args.output
    if not out_path:
        out_path = args.json_file.with_suffix(".html")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print(f"Dashboard saved to: {out_path}")
    print(f"Open in browser: file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
