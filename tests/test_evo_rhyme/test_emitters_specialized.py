from evo_rhyme.emitters import create_emitters


def test_create_emitters_includes_specialized_set():
    emitters, scheduler = create_emitters(
        {
            "theme_keywords": ["pressure", "mask"],
            "min_syllables": 6,
            "max_syllables": 18,
            "schemes": ["AABB", "ABAB"],
            "crossover_rate": 0.6,
        }
    )
    names = {e.name for e in emitters}
    assert {
        "random",
        "internal_rhyme",
        "narrative",
        "punchline",
        "flow",
        "imagery",
        "niche_targeting",
        "repair",
    }.issubset(names)
    assert scheduler is not None

