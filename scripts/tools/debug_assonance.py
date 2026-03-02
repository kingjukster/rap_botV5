#!/usr/bin/env python
"""Debug assonance detection for specific word pairs."""

from rapbot.rhyme_detector import RhymeDetector
from scripts.tools.update_rhyme_groups import extract_last_syllable

d = RhymeDetector()

test_cases = [
    ('pie', 'sky'),
    ('eyes', 'sighs'),
    ('light', 'tonight'),
    ('play', 'day'),
    ('aglow', 'grow'),
]

print("=" * 70)
print("ASSONANCE DEBUG")
print("=" * 70)

for w1, w2 in test_cases:
    f1 = extract_last_syllable(w1)
    f2 = extract_last_syllable(w2)
    r = d.detect_word_rhyme(w1, w2)
    a = d.detect_assonance(w1, w2)
    
    print(f"\n{w1} / {w2}:")
    print(f"  Word rhyme: {r.rhyme_type.value} (sim: {r.similarity:.2f})")
    print(f"  Direct assonance: {a.rhyme_type.value if a else 'None'}")
    if f1 and f2:
        print(f"  Vowels: {f1.vowel} / {f2.vowel}")
        print(f"  Codas: {f1.coda} / {f2.coda}")
        print(f"  Codas match: {f1.coda == f2.coda}")
