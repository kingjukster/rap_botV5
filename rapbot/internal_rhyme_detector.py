"""
internal_rhyme_detector.py

Internal rhyme detection: candidate spans, phonetic signatures, ANN search,
window search (same line + N nearby lines), and rhyme family clustering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from rapbot.phoneme_converter import ARPA_VOWELS, PhonemeSequence, RAP_CONTRACTIONS, strip_stress, word_to_phonemes
from rapbot.rhyme_similarity import classify_rhyme_type, compute_rhyme_similarity, compute_tail_similarity

# Stopwords that are weak rhyme anchors; exclude from internal rhyme detection
RHYME_ANCHOR_STOPWORDS = frozenset({
    "a", "an", "and", "are", "at", "be", "but", "by", "for", "from", "he", "her", "him", "his",
    "i", "id", "im", "i'm", "in", "is", "it", "it's", "me", "my", "no", "of", "on", "or", "our", "out",
    "she", "so", "that", "that's", "the", "them", "then", "there", "they", "they're", "this", "to", "was",
    "we", "we're", "when", "with", "you", "ya", "yo",
    "don't", "can't", "ain't",
    "'em", "'cause", "'bout",
    "em", "cause", "bout",
})

# Try networkx for community detection; fall back to sklearn
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from sklearn.cluster import DBSCAN


WORD_RE = re.compile(r"[A-Za-z']+")

# Short content-word whitelist for rhyme anchors
CONTENT_WORD_WHITELIST = frozenset({"rap", "bar", "flow", "drop", "bars", "beats", "rhyme", "verse"})


def _looks_like_content_word(word: str) -> bool:
    """Return True if word appears content-like (length >= 3 or in whitelist)."""
    w = word.lower()
    return len(w) >= 3 or w in CONTENT_WORD_WHITELIST


@dataclass
class Span:
    """Text span with position metadata."""
    text: str
    start_char: int
    end_char: int
    line_idx: int
    word_indices: Tuple[int, ...]


@dataclass
class SpanTailMeta:
    """Metadata for a span's rhyme tail (from last word for multiword spans)."""
    tail: Tuple[str, ...]
    tail_syllables: int
    has_stressed_tail: bool


@dataclass
class RhymeCluster:
    """Cluster of rhyming spans."""
    family_id: int
    spans: List[Span]
    confidence: float
    rhyme_type: str  # "perfect" | "slant" | "assonance" | "consonance"
    # Canonical family (populated by rhyme_chain_detector):
    representative_tail: Optional[Tuple[str, ...]] = None  # most frequent or longest tail phonemes
    top_anchors: Optional[List[str]] = None  # unigram tokens from span last words
    example_spans: Optional[List[str]] = None  # example span texts


@dataclass
class RepetitionGroup:
    """Repeated surface phrase with span locations."""
    phrase: str
    spans: List[Span]


def _extract_words(text: str) -> List[Tuple[str, int, int]]:
    """Extract (word, start, end) from text."""
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text.lower())]


def _span_has_eligible_anchor(span: Span) -> bool:
    """Return True if span contains >= 1 content word (not stopword)."""
    for w in span.text.lower().split():
        w = w.rstrip(".,!?;:'\"")
        check = RAP_CONTRACTIONS.get(w, w)
        if isinstance(check, str) and " " in check:
            check = check.split()[0]
        if (len(w) >= 3 or w in CONTENT_WORD_WHITELIST) and check not in RHYME_ANCHOR_STOPWORDS:
            return True
    return False


def _compute_cluster_representative_tail(cluster: RhymeCluster) -> Optional[Tuple[str, ...]]:
    """
    Compute representative tail for cluster: most frequent tail among spans,
    or longest on tie. Uses _span_tail_phonemes for consistency with multiword expansion.
    """
    tail_counts: dict[Tuple[str, ...], int] = {}
    for span in cluster.spans:
        tail = _span_tail_phonemes(span)
        if not tail:
            continue
        tail_counts[tail] = tail_counts.get(tail, 0) + 1
    if not tail_counts:
        return None
    max_count = max(tail_counts.values())
    candidates = [t for t, c in tail_counts.items() if c == max_count]
    return max(candidates, key=len)


def _span_tail_phonemes(span: Span) -> Tuple[str, ...]:
    """
    Get the tail phonemes of a multi-word span: concatenate phonemes of all words,
    then extract from the last stressed vowel to end.
    Returns empty tuple if any word has no phonemes or no stressed vowel.
    """
    words = span.text.split()
    if not words:
        return ()
    all_phones: List[str] = []
    last_stressed_idx: Optional[int] = None
    offset = 0
    for w in words:
        seq = word_to_phonemes(w)
        if not seq:
            return ()
        all_phones.extend(seq.phonemes)
        if seq.stressed_vowels:
            last_stressed_idx = offset + seq.stressed_vowels[-1]
        offset += len(seq.phonemes)
    if last_stressed_idx is None:
        return ()
    return tuple(all_phones[last_stressed_idx:])


def _span_tail_meta(span: Span) -> Optional[SpanTailMeta]:
    """
    Get tail metadata for a span. For multiword spans, use last word's tail and stress.
    Returns None if any word has no phonemes.
    """
    words = span.text.split()
    if not words:
        return None
    last_word = words[-1].lower().rstrip(".,!?;:'\"")
    last_seq = word_to_phonemes(last_word)
    if not last_seq or not last_seq.rhyme_nucleus:
        return None
    tail_syl = sum(1 for p in last_seq.rhyme_nucleus if strip_stress(p) in ARPA_VOWELS)
    return SpanTailMeta(
        tail=last_seq.rhyme_nucleus,
        tail_syllables=tail_syl,
        has_stressed_tail=last_seq.has_stressed_rhyme_vowel,
    )


def _generate_candidate_spans(
    lines: List[str],
    min_words: int = 1,
    max_words: int = 1,
    allow_phonetic_bigrams: bool = False,
) -> List[Span]:
    """
    Generate candidate spans for rhyme clustering. Default: unigrams only (min_words=1, max_words=1).

    When allow_phonetic_bigrams=True, also generate bigrams BUT only if:
    (a) at least one content word (existing rule), AND
    (b) the combined span tail (from last stressed vowel) is >= max of both single-word tails.
    This allows "stools in 'em" (tail similar to Jerusalem) but not "and can't hold".
    """
    spans: List[Span] = []
    for line_idx, line in enumerate(lines):
        words_with_pos = _extract_words(line)
        max_n = max_words if not allow_phonetic_bigrams else max(max_words, 2)
        max_n = min(max_n, len(words_with_pos))
        for n in range(min_words, max_n + 1):
            for i in range(len(words_with_pos) - n + 1):
                chunk = words_with_pos[i : i + n]
                if n >= 2:
                    # Require at least one content-like word (length >= 3, not stopword)
                    has_content = False
                    for w_tuple in chunk:
                        w = w_tuple[0].lower().rstrip(".,!?;:'\"")
                        check = RAP_CONTRACTIONS.get(w, w)
                        if isinstance(check, str) and " " in check:
                            check = check.split()[0]
                        if (len(w) >= 3 or w in CONTENT_WORD_WHITELIST) and check not in RHYME_ANCHOR_STOPWORDS:
                            has_content = True
                            break
                    if not has_content:
                        continue
                    # For bigrams with allow_phonetic_bigrams: require span tail >= max of single-word tails
                    if n == 2 and allow_phonetic_bigrams:
                        span_obj = Span(
                            text=" ".join(w[0] for w in chunk),
                            start_char=chunk[0][1],
                            end_char=chunk[-1][2],
                            line_idx=line_idx,
                            word_indices=tuple(range(i, i + n)),
                        )
                        span_tail = _span_tail_phonemes(span_obj)
                        w1_span = Span(chunk[0][0], chunk[0][1], chunk[0][2], line_idx, (i,))
                        w2_span = Span(chunk[1][0], chunk[1][1], chunk[1][2], line_idx, (i + 1,))
                        tail1 = _span_tail_phonemes(w1_span)
                        tail2 = _span_tail_phonemes(w2_span)
                        if len(span_tail) < max(len(tail1), len(tail2)):
                            continue
                        # When last word is stopword/contraction-like: require stressed tail + tail_phones >= 3
                        last_w = chunk[-1][0].lower().rstrip(".,!?;:'\"")
                        check_last = RAP_CONTRACTIONS.get(last_w, last_w)
                        if isinstance(check_last, str) and " " in check_last:
                            check_last = check_last.split()[0]
                        if last_w in RHYME_ANCHOR_STOPWORDS or (len(last_w) < 3 and check_last in RHYME_ANCHOR_STOPWORDS):
                            meta = _span_tail_meta(span_obj)
                            if not meta or not meta.has_stressed_tail or len(meta.tail) < 3:
                                continue
                text = " ".join(w[0] for w in chunk)
                start_char = chunk[0][1]
                end_char = chunk[-1][2]
                word_indices = tuple(range(i, i + n))
                spans.append(Span(
                    text=text,
                    start_char=start_char,
                    end_char=end_char,
                    line_idx=line_idx,
                    word_indices=word_indices,
                ))
    return spans


def detect_repetitions(lines: List[str]) -> List[RepetitionGroup]:
    """
    Find repeated surface n-grams (bigrams, trigrams) across lines.
    Returns list of RepetitionGroup(phrase, spans). Do NOT feed into rhyme clustering.
    """
    phrase_to_spans: dict = {}
    for line_idx, line in enumerate(lines):
        words_with_pos = _extract_words(line)
        for n in (2, 3):  # bigrams, trigrams
            for i in range(len(words_with_pos) - n + 1):
                chunk = words_with_pos[i : i + n]
                phrase = " ".join(w[0] for w in chunk)
                phrase_lower = phrase.lower()
                start_char = chunk[0][1]
                end_char = chunk[-1][2]
                span = Span(
                    text=phrase,
                    start_char=start_char,
                    end_char=end_char,
                    line_idx=line_idx,
                    word_indices=tuple(range(i, i + n)),
                )
                phrase_to_spans.setdefault(phrase_lower, []).append(span)
    return [
        RepetitionGroup(phrase=phrase, spans=spans)
        for phrase, spans in phrase_to_spans.items()
        if len(spans) >= 2
    ]


def _span_phonetic_signature(span: Span) -> Optional[Tuple[Tuple[str, ...], Optional[PhonemeSequence]]]:
    """
    Compute phonetic signature for a span.
    Returns (tuple of phoneme strings, PhonemeSequence or None).
    """
    words = span.text.split()
    all_phones: List[str] = []
    last_seq: Optional[PhonemeSequence] = None
    for w in words:
        seq = word_to_phonemes(w)
        if seq:
            all_phones.extend(seq.phonemes)
            last_seq = seq
        else:
            return None
    if not all_phones:
        return None
    return (tuple(all_phones), last_seq)


def _is_valid_rhyme_anchor(span: Span, phoneme_seq: Optional[PhonemeSequence]) -> bool:
    """
    Return True only if the rhyme-carrying (last) word is a valid rhyme anchor:
    - Has stress on vowel (stressed_vowels non-empty)
    - >= 2 phonemes in the word
    - Not in stopword list (with contraction expansion)
    - Tail length: rhyme_nucleus >= 3 phonemes OR >= 2 syllables
    - Last word is content-like (length >= 3 or in whitelist)
    """
    if not phoneme_seq:
        return False
    words = span.text.lower().split()
    if not words:
        return False
    last_word_raw = words[-1]
    last_word = last_word_raw.lower().rstrip(".,!?;:'\"")
    # Expand contractions for stopword check
    check_word = RAP_CONTRACTIONS.get(last_word, last_word)
    if isinstance(check_word, str) and " " in check_word:
        check_word = check_word.split()[0]  # e.g. "going to" -> "going"
    if check_word in RHYME_ANCHOR_STOPWORDS:
        return False
    if len(phoneme_seq.phonemes) < 2:
        return False
    if not phoneme_seq.stressed_vowels:
        return False
    # Tail length: nucleus >= 3 phonemes OR >= 2 syllables
    tail_ok = len(phoneme_seq.rhyme_nucleus) >= 3 or len(phoneme_seq.syllable_boundaries) >= 2
    if not tail_ok:
        return False
    # Last word must be content-like
    if not _looks_like_content_word(last_word):
        return False
    return True


def _signature_vector(phones: Tuple[str, ...], vocab: dict, dim: int = 64) -> np.ndarray:
    """
    Map phoneme sequence to fixed-dim vector for ANN.
    Simple hash-based embedding for fast lookup.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for i, p in enumerate(phones[-8:]):  # Last 8 phonemes matter most for rhyme
        h = hash(p) % dim
        vec[h] += 1.0 / (i + 1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _build_span_vectors(
    spans: List[Span],
) -> Tuple[np.ndarray, List[Tuple[Span, Tuple[str, ...], PhonemeSequence]]]:
    """Build vector representation for each span with valid phonemes and valid rhyme anchor."""
    vectors = []
    valid: List[Tuple[Span, Tuple[str, ...], PhonemeSequence]] = []
    for span in spans:
        sig = _span_phonetic_signature(span)
        if sig:
            phones, last_seq = sig
            if _is_valid_rhyme_anchor(span, last_seq):
                vec = _signature_vector(phones, {}, 64)
                vectors.append(vec)
                valid.append((span, phones, last_seq))
    if not vectors:
        return np.zeros((0, 64)), []
    return np.stack(vectors), valid


def _window_search(
    lines: List[str],
    line_idx: int,
    window_lines: int = 2,
) -> List[str]:
    """Return lines in window: same line + N nearby lines."""
    lo = max(0, line_idx - window_lines)
    hi = min(len(lines), line_idx + window_lines + 1)
    return lines[lo:hi]


def _expand_clusters_with_multiword_spans(
    clusters: List[RhymeCluster],
    lines: List[str],
    similarity_threshold: float,
) -> None:
    """
    Post-cluster expansion: add 2-3 word spans that bridge to cluster's representative tail.
    Rules:
    1. Span contains >= 1 eligible anchor word (content word, not stopword)
    2. Combined tail length (phonemes) > best single-word tail in the span
    3. Similarity to cluster's representative_tail >= similarity_threshold
    """
    for cluster in clusters:
        rep_tail = _compute_cluster_representative_tail(cluster)
        if not rep_tail:
            continue

        for line_idx, line in enumerate(lines):
            words_with_pos = _extract_words(line)
            for n in (2, 3):  # 2-3 word spans only
                for i in range(len(words_with_pos) - n + 1):
                    chunk = words_with_pos[i : i + n]
                    span = Span(
                        text=" ".join(w[0] for w in chunk),
                        start_char=chunk[0][1],
                        end_char=chunk[-1][2],
                        line_idx=line_idx,
                        word_indices=tuple(range(i, i + n)),
                    )
                    if not _span_has_eligible_anchor(span):
                        continue
                    # Do not add spans whose last word is a stopword (e.g. "song and", "womb and")
                    last_w = chunk[-1][0].lower().rstrip(".,!?;:'\"")
                    check_w = RAP_CONTRACTIONS.get(last_w, last_w)
                    if isinstance(check_w, str) and " " in check_w:
                        check_w = check_w.split()[0]
                    if check_w in RHYME_ANCHOR_STOPWORDS:
                        continue
                    span_tail = _span_tail_phonemes(span)
                    if not span_tail:
                        continue
                    # Max single-word tail length in span
                    max_single_tail = 0
                    for j, w_tuple in enumerate(chunk):
                        w_span = Span(
                            w_tuple[0], w_tuple[1], w_tuple[2],
                            line_idx, (i + j,),
                        )
                        wt = _span_tail_phonemes(w_span)
                        max_single_tail = max(max_single_tail, len(wt))
                    if len(span_tail) <= max_single_tail:
                        continue
                    # Use compute_tail_similarity: both are already tails (no re-extraction)
                    if compute_tail_similarity(
                        list(span_tail), list(rep_tail),
                        min_tail_phonemes=2,
                    ) < similarity_threshold:
                        continue
                    cluster.spans.append(span)


def detect_internal_rhymes(
    lines: List[str],
    similarity_threshold: float = 0.72,
    window_lines: int = 2,
    min_cluster_size: int = 2,
    allow_phonetic_bigrams: bool = False,
) -> List[RhymeCluster]:
    """
    Detect internal rhymes across lines using ANN and clustering.

    Args:
        lines: Verse lines
        similarity_threshold: Min similarity to consider a rhyme
        window_lines: Search within same line ± N lines
        min_cluster_size: Min spans per cluster
        allow_phonetic_bigrams: When True, include bigrams whose combined tail is
            >= max of both single-word tails (e.g. "stools in 'em" ~ Jerusalem)

    Returns:
        List of RhymeCluster
    """
    spans = _generate_candidate_spans(lines, allow_phonetic_bigrams=allow_phonetic_bigrams)
    if len(spans) < 2:
        return []

    vectors, valid = _build_span_vectors(spans)
    if len(valid) < 2:
        return []

    # ANN: sklearn NearestNeighbors (faiss fallback not needed for verse-scale)
    n_neighbors = min(20, len(valid))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(vectors)

    # Build similarity graph
    # min_tail_phonemes_by_type: assonance >= 2 syllables; perfect/slant >= 3 phonemes
    def _tail_ok_for_type(seq: PhonemeSequence, rhyme_type: str) -> bool:
        if rhyme_type == "assonance":
            return len(seq.syllable_boundaries) >= 2
        return len(seq.rhyme_tail(2)) >= 3  # perfect, slant, consonance

    edges: List[Tuple[int, int, float]] = []

    for i, (span_i, phones_i, last_seq_i) in enumerate(valid):
        # Only compare within window
        window = _window_search(lines, span_i.line_idx, window_lines)
        # Get neighbors
        vec = vectors[i : i + 1]
        dists, idxs = nn.kneighbors(vec, n_neighbors=n_neighbors)
        for j, (dist, idx) in enumerate(zip(dists[0], idxs[0])):
            if idx == i:
                continue
            span_j, phones_j, last_seq_j = valid[idx]
            if span_j.line_idx < span_i.line_idx - window_lines or span_j.line_idx > span_i.line_idx + window_lines:
                continue
            sim = 1.0 - dist
            # Use last_seq (PhonemeSequence) for rhyme functions - raw phones use fallback
            # that can truncate to "last vowel" (unstressed) giving wrong tail.
            both_stressed = last_seq_i.has_stressed_rhyme_vowel and last_seq_j.has_stressed_rhyme_vowel
            thr = 0.70 if both_stressed else 0.78
            # Extra guard for unstressed: require longer tail to avoid "you/to/do" clusters
            if not both_stressed:
                tail_phones_i = len(last_seq_i.rhyme_nucleus)
                tail_phones_j = len(last_seq_j.rhyme_nucleus)
                tail_syl_i = sum(1 for p in last_seq_i.rhyme_nucleus if strip_stress(p) in ARPA_VOWELS)
                tail_syl_j = sum(1 for p in last_seq_j.rhyme_nucleus if strip_stress(p) in ARPA_VOWELS)
                unstressed_ok = (
                    (tail_phones_i >= 4 or tail_syl_i >= 2) and
                    (tail_phones_j >= 4 or tail_syl_j >= 2)
                )
                if not unstressed_ok:
                    continue
            ph_sim = compute_rhyme_similarity(
                last_seq_i, last_seq_j, tail_syllables=1, min_tail_phonemes=2
            )
            if sim >= thr and ph_sim >= thr:
                rhyme_type = classify_rhyme_type(
                    last_seq_i, last_seq_j, tail_syllables=1,
                    has_stressed_tail1=last_seq_i.has_stressed_rhyme_vowel,
                    has_stressed_tail2=last_seq_j.has_stressed_rhyme_vowel,
                )
                # Skip edge if weak-tail or tail too short for this rhyme type
                tail_ok_i = _tail_ok_for_type(last_seq_i, rhyme_type)
                tail_ok_j = _tail_ok_for_type(last_seq_j, rhyme_type)
                if rhyme_type in (None, "weak-tail") or not tail_ok_i or not tail_ok_j:
                    continue
                edges.append((i, idx, (sim + ph_sim) / 2, rhyme_type))

    # Deduplicate edges by (i,j) and type; keep max weight per (i,j,type)
    # Group edges by rhyme_type for type-separate clustering
    edge_weights_by_type: dict = {}
    for edge in edges:
        i, j, w = edge[0], edge[1], edge[2]
        rt = edge[3]
        if i > j:
            i, j = j, i
        key = (i, j)
        d = edge_weights_by_type.setdefault(rt, {})
        d[key] = max(d.get(key, 0), w)

    # Cluster separately per rhyme type; collect (nodes, rhyme_type) for each community
    communities_with_type: List[Tuple[Set[int], str]] = []
    for rhyme_type, edge_weights in edge_weights_by_type.items():
        if HAS_NETWORKX and edge_weights:
            G = nx.Graph()
            for (i, j), w in edge_weights.items():
                G.add_edge(i, j, weight=w)
            try:
                from networkx.algorithms import community
                communities = list(community.greedy_modularity_communities(G))
            except Exception:
                communities = list(nx.connected_components(G))
            for comm in communities:
                communities_with_type.append((comm, rhyme_type))
        else:
            # DBSCAN on vectors with distance matrix (1 - similarity)
            n = len(valid)
            dist_mat = np.ones((n, n))
            for i in range(n):
                dist_mat[i, i] = 0
            for (i, j), w in edge_weights.items():
                d = 1.0 - w
                dist_mat[i, j] = dist_mat[j, i] = d
            eps = 1.0 - similarity_threshold
            db = DBSCAN(eps=eps, min_samples=min_cluster_size, metric="precomputed")
            labels = db.fit_predict(dist_mat)
            label_to_nodes: dict = {}
            for idx, lab in enumerate(labels):
                if lab >= 0:
                    label_to_nodes.setdefault(lab, set()).add(idx)
            for comm in [label_to_nodes[k] for k in label_to_nodes]:
                communities_with_type.append((comm, rhyme_type))

    clusters: List[RhymeCluster] = []
    for fam_id, (nodes, cluster_rhyme_type) in enumerate(communities_with_type):
        if len(nodes) < min_cluster_size:
            continue
        cluster_spans = [valid[i][0] for i in nodes]
        sims = []
        for i in nodes:
            for j in nodes:
                if i < j:
                    _, _, seq_i = valid[i]
                    _, _, seq_j = valid[j]
                    sims.append(compute_rhyme_similarity(
                        seq_i, seq_j, tail_syllables=2, min_tail_phonemes=2
                    ))
        confidence = float(np.mean(sims)) if sims else 0.6
        clusters.append(RhymeCluster(
            family_id=fam_id,
            spans=cluster_spans,
            confidence=round(confidence, 4),
            rhyme_type=cluster_rhyme_type,
        ))

    _expand_clusters_with_multiword_spans(clusters, lines, similarity_threshold)
    return clusters
