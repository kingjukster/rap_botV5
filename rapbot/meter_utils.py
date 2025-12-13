# meter_utils.py
import re
import pronouncing
from math import exp

WORD_RE = re.compile(r"[A-Za-z']+")


def estimate_syllables_in_word(word: str) -> int:
    """
    Use CMUdict via pronouncing; fall back to a rough heuristic if OOV.
    """
    word = word.lower()
    phones_list = pronouncing.phones_for_word(word)
    if phones_list:
        # Take first pronunciation, count digits in phones string
        phones = phones_list[0]
        return sum(ch.isdigit() for ch in phones)

    # Fallback heuristic: count vowel groups
    # This is rough but adequate as a backup.
    return max(1, len(re.findall(r"[aeiouy]+", word)))


def count_syllables_in_bar(bar: str) -> int:
    words = WORD_RE.findall(bar.lower())
    return sum(estimate_syllables_in_word(w) for w in words)


def meter_score(
    bar: str,
    target_syllables: int = 13,
    sigma: float = 2.0,
) -> float:
    """
    Returns a score in (0,1] where 1 is 'perfect' match on target syllables.

    Using a Gaussian around target_syllables:
      score = exp(- (s - target)^2 / (2 sigma^2))
    """
    s = count_syllables_in_bar(bar)
    return float(exp(-((s - target_syllables) ** 2) / (2.0 * sigma**2)))
