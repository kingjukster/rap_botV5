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

# LM proposer
try:
    from evo_rhyme.lm_proposer import BarProposer, ProposerConfig, BarRequest
except ImportError:
    pass

# LM rewriter
try:
    from evo_rhyme.lm_rewriter import BarRewriter, RewriterConfig
except ImportError:
    pass

# MAP-Elites archive
try:
    from evo_rhyme.archive import (
        MAPElitesArchive,
        ArchiveDimension,
        create_verse_archive,
        default_verse_dimensions,
    )
except ImportError:
    pass

# Scoring – coherence
try:
    from evo_rhyme.scoring.coherence import score_coherence, score_coherence_with_embeddings
except ImportError:
    pass

# Scoring – punchline
try:
    from evo_rhyme.scoring.punchline import score_punchline, score_punchline_per_line
except ImportError:
    pass

# Multi-objective fitness utilities
try:
    from evo_rhyme.fitness import score_vector, OBJECTIVE_KEYS
except ImportError:
    pass

# Pareto / NSGA-II selection utilities
try:
    from evo_rhyme.selection import (
        pareto_rank,
        crowding_distance,
        pareto_tournament_select,
        pareto_elitism,
        compute_population_objectives,
    )
except ImportError:
    pass

# New individual types
try:
    from evo_rhyme.individual import VerseStructure, create_verse_individual
except ImportError:
    pass

# QD evolution
try:
    from evo_rhyme.verse_evolution import evolve_verse_qd, QDEvolutionConfig
except ImportError:
    pass

# LM population seeder
try:
    from evo_rhyme.population import LMVerseSeedGenerator
except ImportError:
    pass

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
    # LM proposer
    "BarProposer",
    "ProposerConfig",
    "BarRequest",
    # LM rewriter
    "BarRewriter",
    "RewriterConfig",
    # MAP-Elites archive
    "MAPElitesArchive",
    "ArchiveDimension",
    "create_verse_archive",
    "default_verse_dimensions",
    # Scoring
    "score_coherence",
    "score_coherence_with_embeddings",
    "score_punchline",
    "score_punchline_per_line",
    # Multi-objective fitness
    "score_vector",
    "OBJECTIVE_KEYS",
    # Selection
    "pareto_rank",
    "crowding_distance",
    "pareto_tournament_select",
    "pareto_elitism",
    "compute_population_objectives",
    # Individual
    "VerseStructure",
    "create_verse_individual",
    # QD evolution
    "evolve_verse_qd",
    "QDEvolutionConfig",
    # LM population
    "LMVerseSeedGenerator",
]
