# topic_utils.py
from gensim.models import Word2Vec
from pathlib import Path
import numpy as np
import re

WORD_RE = re.compile(r"[A-Za-z']+")


class TopicScorer:
    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Topic model not found: {model_path}")
        self.model: Word2Vec = Word2Vec.load(model_path)

    def _bar_vector(self, text: str):
        tokens = [w.lower() for w in WORD_RE.findall(text)]
        vecs = [self.model.wv[w] for w in tokens if w in self.model.wv]
        if not vecs:
            return None
        return np.mean(vecs, axis=0)

    def similarity(self, bar: str, seed_or_topic: str) -> float:
        """
        Cosine similarity between embeddings of bar and topic words.
        Returns 0 if one side has no in-vocab tokens.
        """
        v1 = self._bar_vector(bar)
        v2 = self._bar_vector(seed_or_topic)
        if v1 is None or v2 is None:
            return 0.0
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)
