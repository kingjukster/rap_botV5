def test_verse_qd_run_logger_writes_acceptance_rate_to_db(monkeypatch):
    """
    VerseQDRunLogger should pass acceptance_rate through to db.insert_generation
    so generations.acceptance_rate is populated (not NULL).
    """
    from evo_rhyme.verse_evolution import VerseQDRunLogger
    import evo_rhyme

    calls = {}

    class _DBStub:
        @staticmethod
        def db_enabled() -> bool:
            return True

        @staticmethod
        def insert_generation(run_id, gen, best_fitness, avg_fitness, diversity, acceptance_rate, extra_json):
            calls["args"] = {
                "run_id": run_id,
                "gen": gen,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "diversity": diversity,
                "acceptance_rate": acceptance_rate,
                "extra_json": extra_json,
            }

    monkeypatch.setattr(evo_rhyme, "db", _DBStub, raising=True)

    logger = VerseQDRunLogger(run_dir=None, run_id=123)
    logger.log_generation(
        gen=7,
        archive_coverage=0.25,
        best_fitness=0.6,
        mean_fitness=0.4,
        occupied_niches=42,
        acceptance_rate=0.33,
        runtime={"gen_wall_time_s": 0.5},
        archive=None,
    )

    assert calls["args"]["run_id"] == 123
    assert calls["args"]["gen"] == 7
    assert calls["args"]["acceptance_rate"] == 0.33
