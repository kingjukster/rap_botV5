# train_topic_embeddings.py
from gensim.models import Word2Vec
from pathlib import Path
import re

WORD_RE = re.compile(r"[A-Za-z']+")


def tokenize_line(line: str):
    return [w.lower() for w in WORD_RE.findall(line)]


def load_corpus(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            # you may want to skip special tokens like [BAR], [RHY=A], etc.
            lines.append(tokenize_line(stripped))
    return lines


def main():
    text_path = "/workspace/rap-botV4/data/elite_kaggle_corpus_clean.txt"
    sentences = load_corpus(text_path)

    model = Word2Vec(
        sentences=sentences,
        vector_size=200,
        window=5,
        min_count=5,
        workers=4,
        sg=1,  # skip-gram
    )
    model.save("/workspace/rap-botV4/elite_w2v.model")
    print("Saved Word2Vec model.")


if __name__ == "__main__":
    main()
