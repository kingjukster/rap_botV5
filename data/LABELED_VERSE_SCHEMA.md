# Labeled Verse Schema

Schema for gold-labeled verses used by `evaluate_verse_analysis.py`.

## Per-item format (JSONL line or JSON object)

```json
{
  "verse": "raw verse text",
  "lines": ["line1", "line2", ...],
  "labels": {
    "rhyme_spans": [[line_idx, start_word_idx, end_word_idx], ...],
    "rhyme_families": [[span_idx1, span_idx2], ...],
    "chains": [[[line_idx, end_word], ...], ...],
    "metaphor_frames": [{"source_domain": "x", "target_domain": "y", "spans": [[line_start, line_end], ...]}],
    "punchlines": [{"line_index": 0, "pivot_word": "..."}]
  },
  "annotators": {"ann1": {...}, "ann2": {...}}
}
```

## Label fields

| Field | Type | Description |
|-------|------|-------------|
| `rhyme_spans` | `[[line, start_w, end_w], ...]` | Gold rhyme spans. Each span: (line_idx, start_word_idx, end_word_idx). Single word: end_w = start_w. |
| `rhyme_families` | `[[span_idx, ...], ...]` | Gold clusters. Indices into `rhyme_spans`. Each sublist = one rhyme family. |
| `chains` | `[[[line_idx, end_word], ...], ...]` | Gold end-rhyme chains. Each chain = list of (line_idx, end_word) positions. |
| `metaphor_frames` | `[{source_domain, target_domain, spans: [[line_start, line_end], ...]}]` | Gold metaphor frames. |
| `punchlines` | `[{line_index, pivot_word}]` | Gold punchlines. `line_index` = 0-based line; `pivot_word` = pivot phrase/word. |

## Multi-annotator (Cohen's kappa)

When `annotators` has 2+ keys, each annotator object may include `punchline_lines` (list of line indices with punchlines) for kappa computation.
