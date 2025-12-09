# train_ngram_critic.py
from ngram_critic import NgramCritic

def main():
    text_path = "/workspace/rap-botV4/data/elite_kaggle_corpus_clean.txt"
    output_path = "/workspace/rap-botV4/elite_ngrams.tsv"
    critic = NgramCritic()
    critic.train_from_corpus(text_path)
    critic.save(output_path)
    print("Saved n-gram critic data:", output_path)

if __name__ == "__main__":
    main()
