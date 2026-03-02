#!/usr/bin/env python
"""
Analyze a verse using the Two-Track Verse Analysis System.
Supports JSON, text, and HTML output formats.
"""

import argparse
import html
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rapbot.internal_rhyme_detector import RHYME_ANCHOR_STOPWORDS, _span_tail_phonemes
from rapbot.phoneme_converter import word_to_phonemes
from rapbot.verse_analyzer import analyze_verse
from rapbot.verse_annotation import VerseAnnotation


def _cluster_phoneme_tail_variants(cluster) -> str:
    """
    Build Phoneme Tail column for a cluster: nucleus + distinct tail variants
    from ALL spans (not just first). Uses _span_tail_phonemes per span.
    Format: "UW1 N | UW1 M" for distinct tails.
    """
    seen: set[tuple[str, ...]] = set()
    variants: list[str] = []
    for span in cluster.spans:
        tail = _span_tail_phonemes(span)
        if not tail or tail in seen:
            continue
        seen.add(tail)
        variants.append(" ".join(tail))
    return " | ".join(variants) if variants else "—"


DEFAULT_VERSE = """You make a wack song, and can't hold a candle
But even Daniel-san wax off, you jack-offs
Need to come to grips, like a hand job
The boom bap is back with an axe to mumble rap
Lumberjack with a hacksaw
Number one, but my pencils are number two 'cause that's all I do with 'em
Poop is my pseudonym
On the john like a prostitute when I droppin' a deuce and
When I'm producing them lyrical bowel movements
These beats are like my saloons
'Cause these bars always got my stools in 'em
And I don't need Metamucil to loosen them
Bitch it is real like I pooped Jerusalem
I'm 'bout to go spin another cocoon
And I'm cuttin' you from your mother's womb and I'm bustin' you."""


def format_text(ann: VerseAnnotation) -> str:
    """Create backward-compatible text output with rhyme highlights + semantics."""
    lines_out = []
    lines_out.append("=" * 80)
    lines_out.append("VERSE ANALYSIS (Two-Track)")
    lines_out.append("=" * 80)
    lines_out.append("")

    # Build rhyme word sets per line from clusters
    rhyme_words_by_line: dict = {}
    for cluster in ann.rhyme_clusters:
        for span in cluster.spans:
            for wi in span.word_indices:
                rhyme_words_by_line.setdefault(span.line_idx, set()).add(
                    span.text.split()[wi] if wi < len(span.text.split()) else span.text
                )
            # Also add whole span text words
            for w in span.text.split():
                rhyme_words_by_line.setdefault(span.line_idx, set()).add(w.lower())

    # End rhyme from chains
    for chain in ann.rhyme_chains:
        for line_idx, word in chain.positions:
            rhyme_words_by_line.setdefault(line_idx, set()).add(word.lower())

    lines_out.append("VERSE WITH HIGHLIGHTS:")
    lines_out.append("-" * 80)
    for i, line in enumerate(ann.lines):
        words = line.split()
        parts = []
        rhyme_set = rhyme_words_by_line.get(i, set())
        for w in words:
            clean = w.rstrip(".,!?;:'\"").lower()
            if clean in rhyme_set:
                parts.append(f"[{w}](RHYME)")
            else:
                parts.append(w)
        lines_out.append(f"Line {i+1:2d}: {' '.join(parts)}")

    lines_out.append("")
    lines_out.append("=" * 80)
    lines_out.append("RHYME STATISTICS")
    lines_out.append("=" * 80)
    lines_out.append(f"Rhyme clusters: {len(ann.rhyme_clusters)}")
    lines_out.append(f"Rhyme chains: {len(ann.rhyme_chains)}")
    for c in ann.rhyme_chains:
        pos_str = ", ".join(f"L{p[0]+1}:{p[1]}" for p in c.positions)
        lines_out.append(f"  Chain {c.chain_id}: {c.pattern} — {pos_str}")
    lines_out.append("")

    # Metaphor frames
    if ann.metaphor_frames:
        lines_out.append("METAPHOR FRAMES:")
        lines_out.append("-" * 80)
        for m in ann.metaphor_frames:
            spans_str = ", ".join(f"L{s[0]+1}-{s[1]+1}" for s in m.spans)
            lines_out.append(f"  {m.source_domain} -> {m.target_domain} ({spans_str}, conf={m.confidence:.2f})")
            if m.cue_words:
                lines_out.append(f"    Cues: {', '.join(m.cue_words)}")
        lines_out.append("")

    # Punchlines
    if ann.punchlines:
        lines_out.append("PUNCHLINES:")
        lines_out.append("-" * 80)
        for p in ann.punchlines:
            lines_out.append(f"  L{p.line_index+1}: \"{p.pivot_word}\" — {p.original_meaning} / {p.double_meaning} (conf={p.confidence:.2f})")
        lines_out.append("")

    # Discourse units
    if ann.discourse_units:
        lines_out.append("DISCOURSE UNITS:")
        lines_out.append("-" * 80)
        for d in ann.discourse_units:
            lines_out.append(f"  {d.type}: lines {d.span[0]+1}-{d.span[1]+1} (conf={d.confidence:.2f})")
        lines_out.append("")

    # Cohesion
    if ann.cohesion_metrics:
        lines_out.append("COHESION METRICS:")
        lines_out.append("-" * 80)
        for k, v in ann.cohesion_metrics.items():
            lines_out.append(f"  {k}: {v}")
        lines_out.append("")

    lines_out.append("CONFIDENCE SCORES:")
    lines_out.append("-" * 80)
    for k, v in ann.confidence_scores.items():
        lines_out.append(f"  {k}: {v:.4f}")
    lines_out.append("")

    return "\n".join(lines_out)


def format_html(ann: VerseAnnotation) -> str:
    """Create dashboard-style HTML output with rhyme, metaphor, and punchline highlights."""
    # Build rhyme word -> cluster mapping for color-coding by family
    CLUSTER_COLORS = [
        "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4",
        "#E91E63", "#8BC34A", "#FF5722", "#673AB7", "#009688",
    ]
    word_to_cluster: dict = {}
    for ci, c in enumerate(ann.rhyme_clusters):
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        for s in c.spans:
            for w in s.text.split():
                word_to_cluster[w.rstrip(".,!?;:'\"").lower()] = (c.family_id, color, c.rhyme_type or "rhyme")
    for c in ann.rhyme_chains:
        for _, w in c.positions:
            word_to_cluster[w.lower()] = (c.family_id, CLUSTER_COLORS[0], "end")

    punchline_words = {p.pivot_word.lower() for p in ann.punchlines}
    metaphor_line_ids = set()
    for m in ann.metaphor_frames:
        for s, e in m.spans:
            for i in range(s, e + 1):
                metaphor_line_ids.add(i)

    # Stance summary
    stance_items = []
    if ann.stance_labels:
        for line_idx, labels in ann.stance_labels.items():
            top = sorted(labels.items(), key=lambda x: -x[1])[:2]
            if top:
                stance_items.extend([(k, v) for k, v in top if v > 0.3])

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Verse Analysis Dashboard</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 24px;
            background: #0f1419;
            color: #e7e9ea;
            line-height: 1.5;
        }
        .container { max-width: 960px; margin: 0 auto; }
        h1 {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1d9bf0;
            margin-bottom: 24px;
        }
        .stats-row {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: #192734;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }
        .stat-value { font-size: 1.75rem; font-weight: 700; color: #1d9bf0; }
        .stat-label { font-size: 0.75rem; color: #71767b; margin-top: 4px; }
        .card {
            background: #192734;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        }
        .card h2 {
            font-size: 0.9rem;
            font-weight: 600;
            color: #1d9bf0;
            margin: 0 0 12px 0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .verse-block {
            background: #15202b;
            border-radius: 12px;
            padding: 20px;
            font-size: 1.05rem;
            line-height: 2;
        }
        .verse-line {
            padding: 6px 0;
            border-left: 3px solid transparent;
            padding-left: 12px;
            margin: 2px 0;
        }
        .verse-line:hover { background: #1c2938; border-radius: 4px; }
        .line-num {
            display: inline-block;
            width: 28px;
            color: #71767b;
            font-size: 0.85rem;
        }
        .rhyme-span {
            padding: 1px 6px;
            border-radius: 4px;
            font-weight: 600;
            cursor: default;
        }
        .punchline-span {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1px 6px;
            border-radius: 4px;
            font-weight: 600;
        }
        .metaphor-line .verse-line { border-left-color: #ff9800; }
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            margin: 2px 4px 2px 0;
        }
        .badge-brag { background: #1d9bf0; color: white; }
        .badge-insult { background: #f44336; color: white; }
        .badge-threat { background: #ff5722; color: white; }
        .meta-card {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 12px;
            background: #15202b;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .meta-card strong { color: #1d9bf0; }
        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 12px;
            font-size: 0.85rem;
        }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .legend-dot { width: 12px; height: 12px; border-radius: 3px; }
        body.show-stopwords .stopword { opacity: 0.4; color: #71767b !important; }
        .cluster-summary-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
        .cluster-summary-table th, .cluster-summary-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #2f3336; }
        .cluster-summary-table th { color: #71767b; font-weight: 500; }
        .cluster-tokens { font-family: monospace; font-size: 0.85rem; color: #8b98a5; }
        .filter-toggle { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; font-size: 0.9rem; }
        .filter-toggle input { cursor: pointer; }
        .filter-toggle label { cursor: pointer; user-select: none; }
    </style>
</head>
<body>
<div class="container">
    <h1>Verse Analysis Dashboard</h1>
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">""" + str(len(ann.lines)) + """</div>
            <div class="stat-label">Lines</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">""" + str(len(ann.rhyme_clusters)) + """</div>
            <div class="stat-label">Rhyme Clusters</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">""" + str(len(ann.rhyme_chains)) + """</div>
            <div class="stat-label">Chains</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">""" + str(len(ann.metaphor_frames)) + """</div>
            <div class="stat-label">Metaphors</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">""" + str(len(ann.punchlines)) + """</div>
            <div class="stat-label">Punchlines</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">""" + (f'{ann.density_metrics.get("overall_technicality_score", 0):.2f}' if ann.density_metrics else '—') + """</div>
            <div class="stat-label">Technicality</div>
        </div>
    </div>
""")

    # Verse with color-coded rhymes
    html_parts.append('<div class="card"><h2>Verse</h2>')
    html_parts.append('''<div class="filter-toggle">
        <input type="checkbox" id="show-stopwords" data-filter="stopwords" aria-label="Show stopwords greyed">
        <label for="show-stopwords">Show stopwords (greyed)</label>
    </div>''')
    html_parts.append('<div class="verse-block' + (' metaphor-line' if metaphor_line_ids else '') + '">')
    for i, line in enumerate(ann.lines):
        words = line.split()
        span_parts = []
        for w in words:
            clean = w.rstrip(".,!?;:'\"").lower()
            stopword_class = " stopword" if clean in RHYME_ANCHOR_STOPWORDS else ""
            if clean in punchline_words:
                span_parts.append(f'<span class="punchline-span{stopword_class}" title="Punchline">{w}</span> ')
            elif clean in word_to_cluster:
                fid, color, rtype = word_to_cluster[clean]
                span_parts.append(f'<span class="rhyme-span{stopword_class}" style="background:{color};color:white" title="Rhyme family {fid} ({rtype})">{w}</span> ')
            else:
                span_parts.append(f'<span class="token{stopword_class}">{w}</span> ')
        html_parts.append(f'<div class="verse-line"><span class="line-num">{i+1:2d}</span>{"".join(span_parts).rstrip()}</div>')
    html_parts.append('</div>')
    html_parts.append('<div class="legend">')
    html_parts.append('<div class="legend-item"><span class="legend-dot" style="background:#f5576c"></span> Punchline</div>')
    for ci, c in enumerate(ann.rhyme_clusters[:5]):
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        html_parts.append(f'<div class="legend-item"><span class="legend-dot" style="background:{color}"></span> Rhyme family {c.family_id}</div>')
    html_parts.append('</div></div>')

    # Metaphor frames
    if ann.metaphor_frames:
        html_parts.append('<div class="card"><h2>Metaphor Frames</h2>')
        for m in ann.metaphor_frames:
            spans_str = ", ".join(f"L{s[0]+1}–{s[1]+1}" for s in m.spans)
            html_parts.append(f'<div class="meta-card">')
            html_parts.append(f'<strong>{m.source_domain}</strong> → <strong>{m.target_domain}</strong>')
            html_parts.append(f'<span style="color:#71767b">({spans_str})</span>')
            if m.cue_words:
                html_parts.append(f'<br><small>Cues: {", ".join(m.cue_words)}</small>')
            html_parts.append('</div>')
        html_parts.append('</div>')

    # Punchlines
    if ann.punchlines:
        html_parts.append('<div class="card"><h2>Punchlines</h2>')
        for p in ann.punchlines:
            html_parts.append(f'<div class="meta-card">')
            html_parts.append(f'<strong>L{p.line_index+1}: {p.pivot_word}</strong><br>')
            html_parts.append(f'<span style="color:#71767b">{p.original_meaning} → {p.double_meaning}</span>')
            html_parts.append('</div>')
        html_parts.append('</div>')

    # Technicality / Density metrics
    if ann.density_metrics:
        dm = ann.density_metrics
        html_parts.append('<div class="card"><h2>Technicality</h2>')
        html_parts.append('<div class="stats-row">')
        for k, v in dm.items():
            if isinstance(v, list):
                v_str = f"mean {sum(v)/len(v):.2f}" if v else "—"
            else:
                v_str = f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
            html_parts.append(f'<div class="stat-card"><div class="stat-value">{v_str}</div><div class="stat-label">{k.replace("_", " ").title()}</div></div>')
        html_parts.append('</div></div>')

    # Rhyme chains
    if ann.rhyme_chains:
        html_parts.append('<div class="card"><h2>Rhyme Chains</h2>')
        for c in ann.rhyme_chains:
            pos_str = ", ".join(f"L{p[0]+1}: {p[1]}" for p in c.positions)
            html_parts.append(f'<div class="meta-card"><strong>{c.pattern}</strong> — {pos_str}</div>')
        html_parts.append('</div>')

    # Cluster summary (after Rhyme Chains)
    if ann.rhyme_clusters:
        html_parts.append('<div class="card"><h2>Cluster Summary</h2>')
        html_parts.append('<table class="cluster-summary-table"><thead><tr><th>Cluster</th><th>Size</th><th>Type</th><th>Phoneme Tail</th><th>Top Tokens</th></tr></thead><tbody>')
        for ci, c in enumerate(ann.rhyme_clusters):
            size = len(c.spans)
            rhyme_type = c.rhyme_type or "rhyme"
            tail = _cluster_phoneme_tail_variants(c)
            top_tokens = ", ".join(s.text for s in c.spans[:5])
            html_parts.append(f'<tr><td>{ci + 1}</td><td>{size}</td><td>{rhyme_type}</td><td class="cluster-tokens">{tail}</td><td class="cluster-tokens">{top_tokens}</td></tr>')
        html_parts.append('</tbody></table></div>')

    # Stance
    top_stance = {}
    for k, v in stance_items:
        top_stance[k] = max(top_stance.get(k, 0), v)
    if top_stance:
        html_parts.append('<div class="card"><h2>Stance</h2>')
        for label, score in sorted(top_stance.items(), key=lambda x: -x[1])[:5]:
            if score > 0.2:
                cls = "badge-brag" if label == "brag" else "badge-insult" if label == "insult" else "badge-threat"
                html_parts.append(f'<span class="badge {cls}">{label} {score:.0%}</span>')
        html_parts.append('</div>')

    html_parts.append('''<script>
        (function() {
            var cb = document.getElementById("show-stopwords");
            if (cb) {
                cb.addEventListener("change", function() {
                    document.body.classList.toggle("show-stopwords", cb.checked);
                });
            }
        })();
    </script>''')
    html_parts.append('</div></body></html>')
    return "".join(html_parts)


def format_html_v4(ann: VerseAnnotation) -> str:
    """Dashboard v4 with Anchor view: anchor_status, source, legend."""
    # Build token lookup by (line_idx, word_idx)
    token_by_pos = {(t.line_idx, t.word_idx): t for t in ann.tokens} if ann.tokens else {}

    CLUSTER_COLORS = [
        "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4",
        "#E91E63", "#8BC34A", "#FF5722", "#673AB7", "#009688",
    ]
    word_to_cluster = {}
    for ci, c in enumerate(ann.rhyme_clusters):
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        for s in c.spans:
            for w in s.text.split():
                word_to_cluster[w.rstrip(".,!?;:'\"").lower()] = (c.family_id, color, c.rhyme_type or "rhyme")
    for c in ann.rhyme_chains:
        for _, w in c.positions:
            word_to_cluster[w.lower()] = (c.family_id, CLUSTER_COLORS[0], "end")

    punchline_words = {p.pivot_word.lower() for p in ann.punchlines}
    metaphor_line_ids = set()
    for m in ann.metaphor_frames:
        for s, e in m.spans:
            for i in range(s, e + 1):
                metaphor_line_ids.add(i)

    stance_items = []
    if ann.stance_labels:
        for line_idx, labels in ann.stance_labels.items():
            top = sorted(labels.items(), key=lambda x: -x[1])[:2]
            if top:
                stance_items.extend([(k, v) for k, v in top if v > 0.3])

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Verse Analysis Dashboard v4</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 24px;
            background: #0f1419;
            color: #e7e9ea;
            line-height: 1.5;
        }
        .container { max-width: 960px; margin: 0 auto; }
        h1 { font-size: 1.5rem; font-weight: 700; color: #1d9bf0; margin-bottom: 24px; }
        .stats-row { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 12px; margin-bottom: 24px; }
        .stat-card { background: #192734; border-radius: 12px; padding: 16px; text-align: center; }
        .stat-value { font-size: 1.75rem; font-weight: 700; color: #1d9bf0; }
        .stat-label { font-size: 0.75rem; color: #71767b; margin-top: 4px; }
        .card { background: #192734; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
        .card h2 { font-size: 0.9rem; font-weight: 600; color: #1d9bf0; margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 0.05em; }
        .verse-block { background: #15202b; border-radius: 12px; padding: 20px; font-size: 1.05rem; line-height: 2; }
        .verse-line { padding: 6px 0; border-left: 3px solid transparent; padding-left: 12px; margin: 2px 0; }
        .verse-line:hover { background: #1c2938; border-radius: 4px; }
        .line-num { display: inline-block; width: 28px; color: #71767b; font-size: 0.85rem; }
        .rhyme-span { padding: 1px 6px; border-radius: 4px; font-weight: 600; cursor: default; }
        .punchline-span { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1px 6px; border-radius: 4px; font-weight: 600; }
        .metaphor-line .verse-line { border-left-color: #ff9800; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 500; margin: 2px 4px 2px 0; }
        .badge-brag { background: #1d9bf0; color: white; }
        .badge-insult { background: #f44336; color: white; }
        .badge-threat { background: #ff5722; color: white; }
        .meta-card { display: flex; align-items: flex-start; gap: 12px; padding: 12px; background: #15202b; border-radius: 8px; margin-bottom: 8px; }
        .meta-card strong { color: #1d9bf0; }
        .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; font-size: 0.85rem; }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .legend-dot { width: 12px; height: 12px; border-radius: 3px; }
        body.show-stopwords .stopword { opacity: 0.4; color: #71767b !important; }
        .eligible { font-weight: 600; }
        .ignored { opacity: 0.85; }
        .weak-tail { border-bottom: none; }
        .oov-g2p { border-bottom: none; }
        body.show-anchor-view .weak-tail { border-bottom: 2px dashed #ff9800; }
        body.show-anchor-view .oov-g2p { border-bottom: 1px dotted #00bcd4; }
        .cluster-summary-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
        .cluster-summary-table th, .cluster-summary-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #2f3336; }
        .cluster-summary-table th { color: #71767b; font-weight: 500; }
        .cluster-tokens { font-family: monospace; font-size: 0.85rem; color: #8b98a5; }
        .filter-toggle { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; font-size: 0.9rem; flex-wrap: wrap; }
        .filter-toggle input { cursor: pointer; }
        .filter-toggle label { cursor: pointer; user-select: none; }
    </style>
</head>
<body>
<div class="container">
    <h1>Verse Analysis Dashboard v4</h1>
    <div class="stats-row">
        <div class="stat-card"><div class="stat-value">""" + str(len(ann.lines)) + """</div><div class="stat-label">Lines</div></div>
        <div class="stat-card"><div class="stat-value">""" + str(len(ann.rhyme_clusters)) + """</div><div class="stat-label">Rhyme Clusters</div></div>
        <div class="stat-card"><div class="stat-value">""" + str(len(ann.rhyme_chains)) + """</div><div class="stat-label">Chains</div></div>
        <div class="stat-card"><div class="stat-value">""" + str(len(ann.metaphor_frames)) + """</div><div class="stat-label">Metaphors</div></div>
        <div class="stat-card"><div class="stat-value">""" + str(len(ann.punchlines)) + """</div><div class="stat-label">Punchlines</div></div>
        <div class="stat-card"><div class="stat-value">""" + (f'{ann.density_metrics.get("overall_technicality_score", 0):.2f}' if ann.density_metrics else '—') + """</div><div class="stat-label">Technicality</div></div>
    </div>
    <div class="card"><h2>Verse</h2>
    <div class="filter-toggle">
        <input type="checkbox" id="show-stopwords" aria-label="Show stopwords greyed">
        <label for="show-stopwords">Show stopwords (greyed)</label>
        <input type="checkbox" id="show-anchor-view" aria-label="Anchor view">
        <label for="show-anchor-view">Anchor view</label>
    </div>
    <div class="verse-block""" + (' metaphor-line' if metaphor_line_ids else '') + '">')

    for i, line in enumerate(ann.lines):
        words = line.split()
        span_parts = []
        for wi, w in enumerate(words):
            clean = w.rstrip(".,!?;:'\"").lower()
            t = token_by_pos.get((i, wi))
            anchor_cls = ""
            if t:
                if t.anchor_status:
                    anchor_cls += " " + t.anchor_status
                if t.source == "g2p":
                    anchor_cls += " oov-g2p"
            # Avoid duplicate stopword: token.anchor_status already includes "stopword"
            stopword_cls = " stopword" if clean in RHYME_ANCHOR_STOPWORDS and (not t or t.anchor_status != "stopword") else ""

            if clean in punchline_words:
                span_parts.append(f'<span class="punchline-span{stopword_cls}{anchor_cls}" style="background:linear-gradient(135deg,#f093fb,#f5576c);color:white" title="Punchline">{w}</span> ')
            elif clean in word_to_cluster:
                fid, color, rtype = word_to_cluster[clean]
                span_parts.append(f'<span class="rhyme-span{stopword_cls}{anchor_cls}" style="background:{color};color:white" title="Rhyme family {fid} ({rtype})">{w}</span> ')
            else:
                span_parts.append(f'<span class="token{stopword_cls}{anchor_cls}">{w}</span> ')

        html_parts.append(f'<div class="verse-line"><span class="line-num">{i+1:2d}</span>{"".join(span_parts).rstrip()}</div>')

    html_parts.append('</div><div class="legend">')
    html_parts.append('<div class="legend-item"><span class="legend-dot" style="background:#f5576c"></span> Punchline</div>')
    for ci, c in enumerate(ann.rhyme_clusters[:5]):
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        html_parts.append(f'<div class="legend-item"><span class="legend-dot" style="background:{color}"></span> Rhyme family {c.family_id}</div>')
    html_parts.append('<div class="legend-item"><span class="legend-dot" style="background:#4CAF50"></span> Eligible (full)</div>')
    html_parts.append('<div class="legend-item"><span class="legend-dot" style="background:#71767b"></span> Stopwords (grey)</div>')
    html_parts.append('<div class="legend-item"><span class="legend-dot" style="background:transparent;border:2px dashed #ff9800"></span> Weak tail (dashed underline)</div>')
    html_parts.append('<div class="legend-item"><span class="legend-dot" style="background:transparent;border:1px dotted #00bcd4"></span> OOV/G2P (dotted underline)</div>')
    html_parts.append('<div class="legend-item">Ignored</div>')
    html_parts.append('</div></div>')

    if ann.metaphor_frames:
        html_parts.append('<div class="card"><h2>Metaphor Frames</h2>')
        for m in ann.metaphor_frames:
            spans_str = ", ".join(f"L{s[0]+1}–{s[1]+1}" for s in m.spans)
            html_parts.append(f'<div class="meta-card"><strong>{m.source_domain}</strong> → <strong>{m.target_domain}</strong> <span style="color:#71767b">({spans_str})</span>')
            if m.cue_words:
                html_parts.append(f'<br><small>Cues: {", ".join(m.cue_words)}</small>')
            html_parts.append('</div>')
        html_parts.append('</div>')

    if ann.punchlines:
        html_parts.append('<div class="card"><h2>Punchlines</h2>')
        for p in ann.punchlines:
            html_parts.append(f'<div class="meta-card"><strong>L{p.line_index+1}: {p.pivot_word}</strong><br><span style="color:#71767b">{p.original_meaning} → {p.double_meaning}</span></div>')
        html_parts.append('</div>')

    if ann.density_metrics:
        dm = ann.density_metrics
        html_parts.append('<div class="card"><h2>Technicality</h2><div class="stats-row">')
        for k, v in dm.items():
            v_str = f"mean {sum(v)/len(v):.2f}" if isinstance(v, list) and v else (f"{v:.2f}" if isinstance(v, (int, float)) else str(v))
            html_parts.append(f'<div class="stat-card"><div class="stat-value">{v_str}</div><div class="stat-label">{k.replace("_", " ").title()}</div></div>')
        html_parts.append('</div></div>')

    if ann.rhyme_chains:
        html_parts.append('<div class="card"><h2>Rhyme Chains</h2>')
        for c in ann.rhyme_chains:
            pos_str = ", ".join(f"L{p[0]+1}: {p[1]}" for p in c.positions)
            html_parts.append(f'<div class="meta-card"><strong>{c.pattern}</strong> — {pos_str}</div>')
        html_parts.append('</div>')

    if ann.rhyme_clusters:
        html_parts.append('<div class="card"><h2>Cluster Summary</h2><table class="cluster-summary-table"><thead><tr><th>Cluster</th><th>Size</th><th>Type</th><th>Phoneme Tail</th><th>Top Tokens</th></tr></thead><tbody>')
        for ci, c in enumerate(ann.rhyme_clusters):
            size = len(c.spans)
            rhyme_type = c.rhyme_type or "rhyme"
            tail = _cluster_phoneme_tail_variants(c)
            top_tokens = ", ".join(s.text for s in c.spans[:5])
            html_parts.append(f'<tr><td>{ci + 1}</td><td>{size}</td><td>{rhyme_type}</td><td class="cluster-tokens">{tail}</td><td class="cluster-tokens">{top_tokens}</td></tr>')
        html_parts.append('</tbody></table></div>')

    top_stance = {}
    for k, v in stance_items:
        top_stance[k] = max(top_stance.get(k, 0), v)
    if top_stance:
        html_parts.append('<div class="card"><h2>Stance</h2>')
        for label, score in sorted(top_stance.items(), key=lambda x: -x[1])[:5]:
            if score > 0.2:
                cls = "badge-brag" if label == "brag" else "badge-insult" if label == "insult" else "badge-threat"
                html_parts.append(f'<span class="badge {cls}">{label} {score:.0%}</span>')
        html_parts.append('</div>')

    html_parts.append('''<script>
        (function() {
            var cbStopwords = document.getElementById("show-stopwords");
            if (cbStopwords) cbStopwords.addEventListener("change", function() { document.body.classList.toggle("show-stopwords", cbStopwords.checked); });
            var cbAnchor = document.getElementById("show-anchor-view");
            if (cbAnchor) cbAnchor.addEventListener("change", function() { document.body.classList.toggle("show-anchor-view", cbAnchor.checked); });
        })();
    </script>''')
    html_parts.append('</div></body></html>')
    return "".join(html_parts)


def format_html_v5(ann: VerseAnnotation) -> str:
    """Dashboard v5 with Chain Inspector, phoneme tooltips on rhyme-spans.
    Renders verse from ann.tokens (Option A) to avoid tokenizer/split() mismatch.
    Uses char_start/char_end to reconstruct gaps (punctuation, hyphens) from original lines.
    """
    CLUSTER_COLORS = [
        "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#00BCD4",
        "#E91E63", "#8BC34A", "#FF5722", "#673AB7", "#009688",
    ]

    punchline_words = {p.pivot_word.lower() for p in ann.punchlines}
    metaphor_line_ids = set()
    for m in ann.metaphor_frames:
        for s, e in m.spans:
            for i in range(s, e + 1):
                metaphor_line_ids.add(i)

    stance_items = []
    if ann.stance_labels:
        for line_idx, labels in ann.stance_labels.items():
            top = sorted(labels.items(), key=lambda x: -x[1])[:2]
            if top:
                stance_items.extend([(k, v) for k, v in top if v > 0.3])

    # Group tokens by line, sorted by char_start (must exist before pos_to_style chain resolution)
    tokens_by_line = {}
    for t in ann.tokens or []:
        tokens_by_line.setdefault(t.line_idx, []).append(t)
    for k in tokens_by_line:
        tokens_by_line[k].sort(key=lambda x: x.char_start)

    pos_to_style = {}  # (line_idx, word_idx) -> (color, rtype, family_id)
    for ci, c in enumerate(ann.rhyme_clusters):
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        rtype = c.rhyme_type or "rhyme"
        fid = str(c.family_id)
        for s in c.spans:
            for wi in s.word_indices:
                pos_to_style[(s.line_idx, wi)] = (color, rtype, fid)
    for c in ann.rhyme_chains:
        color = CLUSTER_COLORS[0]
        fid = str(c.family_id)
        for li, word in c.positions:
            line_tokens = tokens_by_line.get(li, [])
            clean_word = word.rstrip(".,!?;:'\"").lower()
            for t in reversed(line_tokens):
                if t.word.rstrip(".,!?;:'\"").lower() == clean_word:
                    pos_to_style[(li, t.word_idx)] = (color, "chain", fid)
                    break

    PUNCT_CHARS = set('-,.!?;:\'"\'')

    def _phoneme_data(t):
        """Return dict with phonemes, tail, nucleus, source for data-* attrs and tooltip."""
        if not t:
            return None
        word = t.word.rstrip(".,!?;:'\"").lower()
        phonemes = t.phonemes
        tail = t.tail
        nucleus = t.nucleus
        source = t.source or "pronouncing"
        if source == "cmu":
            source = "pronouncing"
        if not phonemes:
            seq = word_to_phonemes(word) if t else None
            if seq:
                phonemes = " ".join(seq.phonemes)
                tail = " ".join(seq.rhyme_nucleus) if seq.rhyme_nucleus else ""
                source = getattr(seq, "source", None) or source
                if source == "cmu":
                    source = "pronouncing"
                for p in (seq.rhyme_nucleus or []):
                    if p and p[-1] in "012":
                        nucleus = p
                        break
        if not phonemes:
            return None
        return {"word": word, "phonemes": phonemes, "tail": tail or "", "nucleus": nucleus or "", "source": source}

    def _phoneme_tooltip(data) -> str:
        """Build canonical tooltip from _phoneme_data result."""
        if not data:
            return ""
        parts = [f"{data['word']} | {data['phonemes']}"]
        if data.get("tail"):
            parts.append(f"tail: {data['tail']}")
        if data.get("nucleus"):
            parts.append(f"nucleus: {data['nucleus']}")
        parts.append(f"source: {data['source']}")
        return " | ".join(parts)

    def _data_attrs(data) -> str:
        """Build data-word, data-phonemes, data-tail, data-nucleus, data-source attributes."""
        if not data:
            return ""
        bits = [f'data-word="{html.escape(data["word"])}"', f'data-phonemes="{html.escape(data["phonemes"])}"',
                f'data-tail="{html.escape(data["tail"])}"', f'data-nucleus="{html.escape(data["nucleus"])}"',
                f'data-source="{html.escape(data["source"])}"']
        return " " + " ".join(bits)

    def _emit_gap(gap: str) -> str:
        """Emit gap: wrap punctuation in spans, escape rest."""
        parts = []
        for c in gap:
            if c in PUNCT_CHARS:
                parts.append(f'<span class="token ignored" title="punctuation">{html.escape(c)}</span>')
            else:
                parts.append(html.escape(c))
        return "".join(parts)

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Verse Analysis Dashboard v7</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 24px;
            background: #0f1419;
            color: #e7e9ea;
            line-height: 1.5;
        }
        .container { max-width: 960px; margin: 0 auto; }
        h1 { font-size: 1.5rem; font-weight: 700; color: #1d9bf0; margin-bottom: 24px; }
        .stats-row { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 12px; margin-bottom: 24px; }
        .stat-card { background: #192734; border-radius: 12px; padding: 16px; text-align: center; }
        .stat-value { font-size: 1.75rem; font-weight: 700; color: #1d9bf0; }
        .stat-label { font-size: 0.75rem; color: #71767b; margin-top: 4px; }
        .card { background: #192734; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
        .card h2 { font-size: 0.9rem; font-weight: 600; color: #1d9bf0; margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 0.05em; }
        .verse-block { background: #15202b; border-radius: 12px; padding: 20px; font-size: 1.05rem; line-height: 2; }
        .verse-line { padding: 6px 0; border-left: 3px solid transparent; padding-left: 12px; margin: 2px 0; }
        .verse-line:hover { background: #1c2938; border-radius: 4px; }
        .line-num { display: inline-block; width: 28px; color: #71767b; font-size: 0.85rem; }
        .token { display: inline-block; padding: 1px 3px; border-radius: 4px; font-weight: 500; cursor: default; }
        .rhyme-span { padding: 1px 6px; border-radius: 4px; font-weight: 600; cursor: default; }
        .punchline-span { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 1px 6px; border-radius: 4px; font-weight: 600; }
        .metaphor-line .verse-line { border-left-color: #ff9800; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 500; margin: 2px 4px 2px 0; }
        .badge-brag { background: #1d9bf0; color: white; }
        .badge-insult { background: #f44336; color: white; }
        .badge-threat { background: #ff5722; color: white; }
        .meta-card { display: flex; align-items: flex-start; gap: 12px; padding: 12px; background: #15202b; border-radius: 8px; margin-bottom: 8px; }
        .meta-card strong { color: #1d9bf0; }
        .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; font-size: 0.85rem; }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .legend-dot { width: 12px; height: 12px; border-radius: 3px; }
        body.show-stopwords .stopword { opacity: 0.4; color: #71767b !important; }
        .eligible { font-weight: 600; outline: 1px solid rgba(255,255,255,0.08); }
        .ignored { opacity: 0.55; }
        .weak-tail { border-bottom: none; }
        .oov-g2p { border-bottom: none; }
        body.show-anchor-view .weak-tail { border-bottom: 2px dashed #ff9800; }
        body.show-anchor-view .oov-g2p { border-bottom: 1px dotted #00bcd4; }
        .cluster-summary-table, .chain-inspector-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
        .cluster-summary-table th, .cluster-summary-table td, .chain-inspector-table th, .chain-inspector-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #2f3336; }
        .cluster-summary-table th, .chain-inspector-table th { color: #71767b; font-weight: 500; }
        .cluster-tokens { font-family: monospace; font-size: 0.85rem; color: #8b98a5; }
        .filter-toggle { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; font-size: 0.9rem; flex-wrap: wrap; }
        .filter-toggle input { cursor: pointer; }
        .filter-toggle label { cursor: pointer; user-select: none; }
        .chain-inspector-chain { margin-bottom: 20px; }
        .chain-inspector-chain:last-child { margin-bottom: 0; }
    </style>
</head>
<body>
<div class="container">
    <h1>Verse Analysis Dashboard v7</h1>
    <div class="stats-row">
        <div class="stat-card"><div class="stat-value">""" + str(len(ann.lines)) + """</div><div class="stat-label">Lines</div></div>
        <div class="stat-card"><div class="stat-value">""" + str(len(ann.rhyme_clusters)) + """</div><div class="stat-label">Rhyme Clusters</div></div>
        <div class="stat-card"><div class="stat-value">""" + str(len(ann.rhyme_chains)) + """</div><div class="stat-label">Chains</div></div>
        <div class="stat-card"><div class="stat-value">""" + str(len(ann.metaphor_frames)) + """</div><div class="stat-label">Metaphors</div></div>
        <div class="stat-card"><div class="stat-value">""" + str(len(ann.punchlines)) + """</div><div class="stat-label">Punchlines</div></div>
        <div class="stat-card"><div class="stat-value">""" + (f'{ann.density_metrics.get("overall_technicality_score", 0):.2f}' if ann.density_metrics else '—') + """</div><div class="stat-label">Technicality</div></div>
    </div>
    <div class="card"><h2>Verse</h2>
    <div class="filter-toggle">
        <input type="checkbox" id="show-stopwords" aria-label="Show stopwords greyed">
        <label for="show-stopwords">Show stopwords (greyed)</label>
        <input type="checkbox" id="show-anchor-view" aria-label="Anchor view">
        <label for="show-anchor-view">Anchor view</label>
    </div>
    <div class="verse-block""" + (' metaphor-line' if metaphor_line_ids else '') + '">')

    verse_text = "\n".join(ann.lines)
    for i, line in enumerate(ann.lines):
        line_start = sum(len(ann.lines[j]) + 1 for j in range(i))
        line_end = line_start + len(line)
        line_tokens = tokens_by_line.get(i, [])
        span_parts = []
        prev = line_start
        for t in line_tokens:
            if __debug__:
                substr = verse_text[t.char_start:t.char_end]
                if substr.lower() != t.word.lower():
                    raise ValueError(f"Token mismatch l{t.line_idx}.w{t.word_idx}: '{substr}' vs '{t.word}'")
            if t.char_start > prev:
                span_parts.append(_emit_gap(verse_text[prev:t.char_start]))
            w = t.word
            clean = w.rstrip(".,!?;:'\"").lower()
            anchor_cls = (" " + t.anchor_status) if t.anchor_status else ""
            if t.source == "g2p":
                anchor_cls += " oov-g2p"
            stopword_cls = " stopword" if clean in RHYME_ANCHOR_STOPWORDS and t.anchor_status != "stopword" else ""
            data = _phoneme_data(t)
            tooltip = _phoneme_tooltip(data)
            data_attrs = _data_attrs(data)
            if clean in punchline_words:
                title = html.escape(tooltip if tooltip else "Punchline")
                span_parts.append(f'<span class="punchline-span{stopword_cls}{anchor_cls}" style="background:linear-gradient(135deg,#f093fb,#f5576c);color:white" title="{title}"{data_attrs}>{html.escape(w)}</span>')
            elif (style := pos_to_style.get((t.line_idx, t.word_idx))):
                color, rtype, fid = style
                title = html.escape(tooltip if tooltip else f"Rhyme family {fid} ({rtype})")
                span_parts.append(f'<span class="rhyme-span{stopword_cls}{anchor_cls}" style="background:{color};color:white" title="{title}"{data_attrs}>{html.escape(w)}</span>')
            else:
                title_attr = f' title="{html.escape(tooltip)}"' if tooltip else ""
                span_parts.append(f'<span class="token{stopword_cls}{anchor_cls}"{title_attr}{data_attrs}>{html.escape(w)}</span>')
            prev = t.char_end
        if prev < line_end:
            span_parts.append(_emit_gap(verse_text[prev:line_end]))
        html_parts.append(f'<div class="verse-line"><span class="line-num">{i+1:2d}</span>{"".join(span_parts)}</div>')

    html_parts.append('</div><div class="legend">')
    html_parts.append('<div class="legend-item"><span class="legend-dot" style="background:#f5576c"></span> Punchline</div>')
    for ci, c in enumerate(ann.rhyme_clusters[:5]):
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        html_parts.append(f'<div class="legend-item"><span class="legend-dot" style="background:{color}"></span> Rhyme family {c.family_id}</div>')
    html_parts.append('<div class="legend-item"><span class="legend-dot" style="background:#4CAF50"></span> Eligible (full)</div>')
    html_parts.append('<div class="legend-item"><span class="legend-dot" style="background:#71767b"></span> Stopwords (grey)</div>')
    html_parts.append('<div class="legend-item"><span class="legend-dot" style="background:transparent;border:2px dashed #ff9800"></span> Weak tail (dashed underline)</div>')
    html_parts.append('<div class="legend-item"><span class="legend-dot" style="background:transparent;border:1px dotted #00bcd4"></span> OOV/G2P (dotted underline)</div>')
    html_parts.append('<div class="legend-item">Hover rhyme words for phoneme tooltip</div>')
    html_parts.append('</div></div>')

    # Chain Inspector
    if ann.rhyme_chains:
        html_parts.append('<div class="card"><h2>Chain Inspector</h2>')
        for ci, c in enumerate(ann.rhyme_chains):
            chain_type_label = "TailChain" if getattr(c, "chain_type", "tail") == "tail" else "NucleusChain"
            rep_nuc = getattr(c, "representative_nucleus", None) or "—"
            rep_tail = getattr(c, "representative_tail", None) or "—"
            rep_display = f"Nucleus: {rep_nuc}" if chain_type_label == "NucleusChain" else f"Tail: {rep_tail}"
            html_parts.append(f'<div class="chain-inspector-chain">')
            html_parts.append(f'<div class="meta-card"><strong>Chain {ci + 1}</strong> — {chain_type_label} | Representative: {rep_display}</div>')
            pos_phon = getattr(c, "positions_with_phonemes", None) or []
            if pos_phon:
                html_parts.append('<table class="chain-inspector-table"><thead><tr><th>Line</th><th>Word/Span</th><th>Full Phonemes</th><th>Tail</th><th>Nucleus</th></tr></thead><tbody>')
                for p in pos_phon:
                    li = p.get("line_idx", 0) + 1
                    w = p.get("word", "")
                    ph = p.get("phonemes", "") or "—"
                    tail = p.get("tail", "") or "—"
                    nuc = p.get("nucleus", "") or "—"
                    html_parts.append(f'<tr><td>{li}</td><td class="cluster-tokens">{w}</td><td class="cluster-tokens">{ph}</td><td class="cluster-tokens">{tail}</td><td class="cluster-tokens">{nuc}</td></tr>')
                html_parts.append('</tbody></table>')
            else:
                for line_idx, word in c.positions:
                    html_parts.append(f'<div class="meta-card">L{line_idx + 1}: {word}</div>')
            html_parts.append('</div>')
        html_parts.append('</div>')

    if ann.metaphor_frames:
        html_parts.append('<div class="card"><h2>Metaphor Frames</h2>')
        for m in ann.metaphor_frames:
            spans_str = ", ".join(f"L{s[0]+1}–{s[1]+1}" for s in m.spans)
            html_parts.append(f'<div class="meta-card"><strong>{m.source_domain}</strong> → <strong>{m.target_domain}</strong> <span style="color:#71767b">({spans_str})</span>')
            if m.cue_words:
                html_parts.append(f'<br><small>Cues: {", ".join(m.cue_words)}</small>')
            html_parts.append('</div>')
        html_parts.append('</div>')

    if ann.punchlines:
        html_parts.append('<div class="card"><h2>Punchlines</h2>')
        for p in ann.punchlines:
            html_parts.append(f'<div class="meta-card"><strong>L{p.line_index+1}: {p.pivot_word}</strong><br><span style="color:#71767b">{p.original_meaning} → {p.double_meaning}</span></div>')
        html_parts.append('</div>')

    if ann.density_metrics:
        dm = ann.density_metrics
        html_parts.append('<div class="card"><h2>Technicality</h2><div class="stats-row">')
        for k, v in dm.items():
            v_str = f"mean {sum(v)/len(v):.2f}" if isinstance(v, list) and v else (f"{v:.2f}" if isinstance(v, (int, float)) else str(v))
            html_parts.append(f'<div class="stat-card"><div class="stat-value">{v_str}</div><div class="stat-label">{k.replace("_", " ").title()}</div></div>')
        html_parts.append('</div></div>')

    if ann.rhyme_chains:
        html_parts.append('<div class="card"><h2>Rhyme Chains</h2>')
        for c in ann.rhyme_chains:
            pos_str = ", ".join(f"L{p[0]+1}: {p[1]}" for p in c.positions)
            html_parts.append(f'<div class="meta-card"><strong>{c.pattern}</strong> — {pos_str}</div>')
        html_parts.append('</div>')

    if ann.rhyme_clusters:
        html_parts.append('<div class="card"><h2>Cluster Summary</h2><table class="cluster-summary-table"><thead><tr><th>Cluster</th><th>Size</th><th>Type</th><th>Phoneme Tail</th><th>Top Tokens</th></tr></thead><tbody>')
        for ci, c in enumerate(ann.rhyme_clusters):
            size = len(c.spans)
            rhyme_type = c.rhyme_type or "rhyme"
            tail = _cluster_phoneme_tail_variants(c)
            top_tokens = ", ".join(s.text for s in c.spans[:5])
            html_parts.append(f'<tr><td>{ci + 1}</td><td>{size}</td><td>{rhyme_type}</td><td class="cluster-tokens">{tail}</td><td class="cluster-tokens">{top_tokens}</td></tr>')
        html_parts.append('</tbody></table></div>')

    top_stance = {}
    for k, v in stance_items:
        top_stance[k] = max(top_stance.get(k, 0), v)
    if top_stance:
        html_parts.append('<div class="card"><h2>Stance</h2>')
        for label, score in sorted(top_stance.items(), key=lambda x: -x[1])[:5]:
            if score > 0.2:
                cls = "badge-brag" if label == "brag" else "badge-insult" if label == "insult" else "badge-threat"
                html_parts.append(f'<span class="badge {cls}">{label} {score:.0%}</span>')
        html_parts.append('</div>')

    html_parts.append('''<script>
        (function() {
            var cbStopwords = document.getElementById("show-stopwords");
            if (cbStopwords) cbStopwords.addEventListener("change", function() { document.body.classList.toggle("show-stopwords", cbStopwords.checked); });
            var cbAnchor = document.getElementById("show-anchor-view");
            if (cbAnchor) cbAnchor.addEventListener("change", function() { document.body.classList.toggle("show-anchor-view", cbAnchor.checked); });
        })();
    </script>''')
    html_parts.append('</div></body></html>')
    return "".join(html_parts)


def main():
    parser = argparse.ArgumentParser(description="Analyze verse with Two-Track system")
    parser.add_argument("--verse", "-v", type=str, help="Verse text (or use --file)")
    parser.add_argument("--file", "-f", type=Path, help="Read verse from file")
    parser.add_argument("--format", "-o", choices=["json", "text", "html"], default="text", help="Output format")
    parser.add_argument("--output", type=Path, help="Write to file (default: data/verse_analysis_output.{json,txt,html})")
    parser.add_argument("--v5", action="store_true", help="Use v5 HTML dashboard (Chain Inspector, phoneme tooltips)")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable sentence-transformers")
    args = parser.parse_args()

    verse = args.verse
    if args.file:
        verse = args.file.read_text(encoding="utf-8")
    if not verse:
        verse = DEFAULT_VERSE

    print("Analyzing verse (Two-Track)...")
    ann = analyze_verse(verse, use_embeddings=not args.no_embeddings)

    if args.format == "json":
        out = ann.to_json()
        path = args.output or Path("data/verse_analysis_output.json")
    elif args.format == "html":
        out = format_html_v5(ann) if args.v5 else format_html(ann)
        path = args.output or Path("data/verse_analysis_output_v5.html" if args.v5 else "data/verse_analysis_output.html")
    else:
        out = format_text(ann)
        path = args.output or Path("data/verse_analysis_output.txt")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out, encoding="utf-8")

    print(f"Output saved to: {path}")
    if args.format == "text":
        try:
            print("\n" + out)
        except UnicodeEncodeError:
            print("\n(Output contains unicode; see file for full content)")


if __name__ == "__main__":
    main()
