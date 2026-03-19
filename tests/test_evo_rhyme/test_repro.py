import random


def test_seed_everything_reproducible_random():
    from evo_rhyme.repro import seed_everything

    seed_everything(123)
    a = [random.random() for _ in range(5)]
    seed_everything(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_reproducible_numpy_if_available():
    from evo_rhyme.repro import seed_everything

    try:
        import numpy as np  # type: ignore
    except Exception:
        return

    seed_everything(456)
    a = np.random.rand(5).tolist()
    seed_everything(456)
    b = np.random.rand(5).tolist()
    assert a == b

