# ngram_critic.py
from collections import Counter, defaultdict
import math
import re
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z']+")


def tokenize(line: str):
    return [w.lower() for w in WORD_RE.findall(line)]


class NgramCritic:
    def __init__(self, ngram_path: str = None):
        """
        If ngram_path exists, load; otherwise stay empty.
        """
        self.max_n = 3
        self.ngram_freqs = [defaultdict(int) for _ in range(self.max_n + 1)]
        if ngram_path and Path(ngram_path).exists():
            self._load(ngram_path)

    def _load(self, path: str):
        # Very simple text-based format:
        # n\t"token1 token2 ... tokenn"\tcount
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_str, seq, count_str = line.split("\t")
                n = int(n_str)
                toks = tuple(seq.split(" "))
                count = int(count_str)
                self.ngram_freqs[n][toks] = count

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for n in range(1, self.max_n + 1):
                for seq, cnt in self.ngram_freqs[n].items():
                    seq_str = " ".join(seq)
                    f.write(f"{n}\t{seq_str}\t{cnt}\n")

    def train_from_corpus(self, text_path: str, min_count: int = 3):
        for n in range(1, self.max_n + 1):
            self.ngram_freqs[n].clear()

        with open(text_path, "r", encoding="utf-8") as f:
            for raw in f:
                tokens = tokenize(raw)
                for n in range(1, self.max_n + 1):
                    for i in range(len(tokens) - n + 1):
                        ngram = tuple(tokens[i : i + n])
                        self.ngram_freqs[n][ngram] += 1

        # Prune low-frequency n-grams
        for n in range(1, self.max_n + 1):
            freqs = self.ngram_freqs[n]
            to_del = [ng for ng, c in freqs.items() if c < min_count]
            for ng in to_del:
                del freqs[ng]

    def critic_score(self, bar: str) -> float:
        """
        Returns a normalized score where higher = more 'seen before'
        with respect to n-grams from the elite corpus.
        Very simple heuristic: fraction of n-grams that exist.
        """
        tokens = tokenize(bar)
        if not tokens:
            return 0.0

        scores = []
        for n in range(1, self.max_n + 1):
            if len(tokens) < n:
                continue
            total = 0
            seen = 0
            for i in range(len(tokens) - n + 1):
                total += 1
                ngram = tuple(tokens[i : i + n])
                if ngram in self.ngram_freqs[n]:
                    seen += 1
            if total > 0:
                scores.append(seen / total)

        if not scores:
            return 0.0
        # Average over n=1,2,3
        return float(sum(scores) / len(scores))
