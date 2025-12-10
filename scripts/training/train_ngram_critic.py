import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# train_ngram_critic.py
import argparse
from rapbot.ngram_critic import NgramCritic
from config.settings import load_settings


def parse_args():
    parser = argparse.ArgumentParser(description="Train the lightweight n-gram critic.")
    parser.add_argument(
        "--text_path",
        type=str,
        default=None,
        help="Corpus to build n-gram stats from (defaults to config elite corpus).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the critic TSV (defaults to config ngram output).",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=3,
        help="Minimum frequency threshold per n-gram.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON/YAML config path overriding defaults.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.config)
    text_path = args.text_path or str(settings.elite_corpus_path)
    output_path = args.output_path or str(settings.ngram_output_path)

    critic = NgramCritic()
    critic.train_from_corpus(text_path, min_count=args.min_count)
    critic.save(output_path)
    print("Saved n-gram critic data:", output_path)


if __name__ == "__main__":
    main()
