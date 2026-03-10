"""
evo_rhyme/rhyme_graph.py

Rhyme graph: edges between words that rhyme (multisyllable overlap >= 2).
Used for internal rhyme density scoring and rhyme-graph-expand mutation.
"""

from __future__ import annotations

from typing import List, Tuple

from evo_rhyme.phonetics import extract_rhyme_tail, multisyllable_overlap, tokenize_line


def build_rhyme_graph(words: List[str]) -> List[Tuple[int, int]]:
    """
    Build rhyme graph: for each pair (i, j) with i < j, get extract_rhyme_tail
    for words[i] and words[j]. If multisyllable_overlap(tail1, tail2) >= 2,
    add edge (i, j). Return list of edges.
    """
    edges: List[Tuple[int, int]] = []
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n):
            tail1 = extract_rhyme_tail(words[i])
            tail2 = extract_rhyme_tail(words[j])
            if multisyllable_overlap(tail1, tail2) >= 2:
                edges.append((i, j))
    return edges


def rhyme_graph_density(edges: List, n: int) -> float:
    """Return density: len(edges) / max(1, n*(n-1)//2)."""
    max_edges = max(1, n * (n - 1) // 2)
    return len(edges) / max_edges


def score_line_rhyme_graph(line: str) -> float:
    """
    Tokenize line, build rhyme graph, return density.
    Returns 0.0 for empty or short lines (< 2 words).
    """
    words = tokenize_line(line)
    if len(words) < 2:
        return 0.0
    edges = build_rhyme_graph(words)
    return rhyme_graph_density(edges, len(words))
