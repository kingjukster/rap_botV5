"""Quick sanity check for new rhyme-family and shell penalties."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo_rhyme.fitness import (
    score_couplet,
    compute_fitness,
    _score_rhyme_family_repetition_penalty,
    FITNESS_CAP,
)
from evo_rhyme.individual import CoupletIndividual, analyze_individual

# Nonsense: truck duck fuck in buck (same rhyme family)
bad = CoupletIndividual(
    line1="truck duck fuck in buck luck",
    line2="blunt full of love lets flow high",
)
analyze_individual(bad)
pen = _score_rhyme_family_repetition_penalty(bad)
print("Rhyme-family repetition penalty for truck duck fuck:", pen)
scores = score_couplet(bad)
fit = compute_fitness(scores, individual=bad, population=[bad])
print("Fitness (capped):", fit)

# Good couplet
good = CoupletIndividual(
    line1="I don't think I'm ever gon' heal",
    line2="this flow we're livin' is real",
)
analyze_individual(good)
pen2 = _score_rhyme_family_repetition_penalty(good)
print("Rhyme-family penalty for good couplet:", pen2)
print("FITNESS_CAP:", FITNESS_CAP)
