"""
Efficient internal rhyme graph metrics.

Builds a lightweight rhyme network over token occurrences and reports:
- graph density
- average local clustering coefficient
- normalized longest-chain estimate
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

from evo_rhyme.phonetics import extract_rhyme_tail, tokenize_line


def _tail_to_key(tail: Tuple[str, ...] | None) -> Tuple[str, ...]:
    if not tail:
        return tuple()
    return tuple(str(p) for p in tail if p)


def _tail_similarity(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    if not a or not b:
        return 0.0
    min_len = min(len(a), len(b))
    suffix = 0
    for i in range(1, min_len + 1):
        if a[-i] == b[-i]:
            suffix += 1
        else:
            break
    return suffix / max(len(a), len(b))


def score_rhyme_graph_metrics(
    lines: List[str],
    max_tokens: int = 96,
    sim_threshold: float = 0.45,
    edge_mode: str = "phonetic",
) -> Dict[str, float]:
    nodes: List[Tuple[str, ...]] = []
    node_words: List[str] = []
    for line in lines:
        for tok in tokenize_line(line):
            tail = _tail_to_key(extract_rhyme_tail(tok))
            if not tail:
                continue
            nodes.append(tail)
            node_words.append(tok.lower())
            if len(nodes) >= max_tokens:
                break
        if len(nodes) >= max_tokens:
            break

    n = len(nodes)
    if n < 2:
        return {
            "rhyme_graph_density": 0.0,
            "rhyme_graph_cluster_coeff": 0.0,
            "rhyme_graph_chain_length": 0.0,
        }

    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, tail in enumerate(nodes):
        key = tail[-1] if tail else ""
        buckets[key].append(i)

    adj: Dict[int, Set[int]] = defaultdict(set)
    edge_count = 0
    for idxs in buckets.values():
        m = len(idxs)
        if m < 2:
            continue
        for i in range(m):
            ni = idxs[i]
            for j in range(i + 1, m):
                nj = idxs[j]
                if edge_mode == "embedding":
                    try:
                        from evo_rhyme.rhyme_embedding import word_pair_rhyme_similarity

                        a = node_words[ni]
                        b = node_words[nj]
                        sim = word_pair_rhyme_similarity(a, b)
                    except Exception:
                        sim = _tail_similarity(nodes[ni], nodes[nj])
                else:
                    sim = _tail_similarity(nodes[ni], nodes[nj])
                if sim >= sim_threshold:
                    adj[ni].add(nj)
                    adj[nj].add(ni)
                    edge_count += 1

    possible_edges = n * (n - 1) / 2.0
    density = edge_count / possible_edges if possible_edges > 0 else 0.0

    clustering_vals: List[float] = []
    for v in range(n):
        neighbors = list(adj.get(v, set()))
        k = len(neighbors)
        if k < 2:
            continue
        tri = 0
        possible = k * (k - 1) / 2.0
        for i in range(k):
            a = neighbors[i]
            for j in range(i + 1, k):
                b = neighbors[j]
                if b in adj.get(a, set()):
                    tri += 1
        clustering_vals.append(tri / possible if possible > 0 else 0.0)
    cluster_coeff = sum(clustering_vals) / len(clustering_vals) if clustering_vals else 0.0

    def _bfs_farthest(start: int) -> Tuple[int, int]:
        q = deque([(start, 0)])
        seen = {start}
        far, far_d = start, 0
        while q:
            cur, d = q.popleft()
            if d > far_d:
                far, far_d = cur, d
            for nxt in adj.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, d + 1))
        return far, far_d

    if not adj:
        chain = 0.0
    else:
        seed = next(iter(adj))
        far, _ = _bfs_farthest(seed)
        _, dist = _bfs_farthest(far)
        chain = dist / max(1, n - 1)

    return {
        "rhyme_graph_density": float(max(0.0, min(1.0, density))),
        "rhyme_graph_cluster_coeff": float(max(0.0, min(1.0, cluster_coeff))),
        "rhyme_graph_chain_length": float(max(0.0, min(1.0, chain))),
    }

