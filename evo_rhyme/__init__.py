"""evo_rhyme: Evolutionary rhyme analysis and scoring."""

from evo_rhyme.individual import (
    CoupletIndividual,
    LineFeatures,
    VerseIndividual,
    VerseFeatures,
    analyze_individual,
    analyze_verse_individual,
)
from evo_rhyme.fitness import (
    score_couplet,
    compute_fitness,
    DEFAULT_WEIGHTS,
    score_verse,
    compute_verse_fitness,
    VERSE_DEFAULT_WEIGHTS,
)
from evo_rhyme.constraints import passes_constraints, passes_verse_constraints, ConstraintConfig
from evo_rhyme.population import (
    RandomGenerator,
    TemplateGenerator,
    VerseSeedGenerator,
    VersePopulation,
    create_initial_population,
    create_mixed_population,
    create_initial_verse_population,
)
from evo_rhyme.evolution import evolve, EvolutionConfig
from evo_rhyme.verse_evolution import (
    evolve_verse_population,
    verse_crossover,
    verse_mutate,
    VerseEvolutionConfig,
)
from evo_rhyme.style_profile import StyleProfile, build_style_profile
from evo_rhyme.seed_generator import SeedGenerator
from evo_rhyme.phonetics import (
    PhoneticFeature,
    count_syllables,
    extract_last_syllable,
    extract_rhyme_tail,
    get_pronunciations,
    multisyllable_overlap,
    phonetic_similarity,
    stress_pattern_from_phones,
)
from evo_rhyme.scoring.end_rhyme import score_end_rhyme
from evo_rhyme.mutation import mutate, MUTATION_WEIGHTS, get_tail_to_words
from evo_rhyme.crossover import crossover
from evo_rhyme.selection import elitism, tournament_select, inject_random_immigrants
from evo_rhyme.generator import (
    generate_couplet,
    get_generator,
    generate_seed_couplets,
    load_templates,
    load_vocab,
    get_group_to_words,
)

__all__ = [
    "CoupletIndividual",
    "LineFeatures",
    "VerseIndividual",
    "VerseFeatures",
    "analyze_individual",
    "analyze_verse_individual",
    "score_couplet",
    "score_verse",
    "compute_verse_fitness",
    "VERSE_DEFAULT_WEIGHTS",
    "passes_verse_constraints",
    "evolve_verse_population",
    "verse_crossover",
    "verse_mutate",
    "VerseEvolutionConfig",
    "VerseSeedGenerator",
    "VersePopulation",
    "create_initial_population",
    "create_initial_verse_population",
    "create_mixed_population",
    "TemplateGenerator",
    "RandomGenerator",
    "evolve",
    "EvolutionConfig",
    "StyleProfile",
    "build_style_profile",
    "SeedGenerator",
    "compute_fitness",
    "DEFAULT_WEIGHTS",
    "passes_constraints",
    "ConstraintConfig",
    "PhoneticFeature",
    "count_syllables",
    "extract_last_syllable",
    "extract_rhyme_tail",
    "get_pronunciations",
    "multisyllable_overlap",
    "phonetic_similarity",
    "stress_pattern_from_phones",
    "score_end_rhyme",
    "mutate",
    "MUTATION_WEIGHTS",
    "get_tail_to_words",
    "crossover",
    "elitism",
    "tournament_select",
    "inject_random_immigrants",
    "generate_couplet",
    "get_generator",
    "generate_seed_couplets",
    "load_templates",
    "load_vocab",
    "get_group_to_words",
]
