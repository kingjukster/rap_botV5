"""RunPod Serverless handler for evolution workers.

Receives a run config dict via the job input and executes run_verse_qd.main()
with synthesized CLI arguments. DB and LLM env vars are set in the RunPod
endpoint environment at dispatch time.
"""

import logging
import sys
import os

import runpod

logger = logging.getLogger("runpod_handler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _build_argv(cfg: dict) -> list:
    """Convert a run config dict into sys.argv for run_verse_qd.parse_args()."""
    argv = ["run_verse_qd.py"]
    argv.extend(["--theme", str(cfg["theme"])])
    argv.extend(["--population", str(cfg.get("population", 80))])
    argv.extend(["--generations", str(cfg.get("generations", 40))])
    argv.extend(["--scheme", str(cfg.get("scheme", "AABB"))])
    argv.extend(["--init", str(cfg.get("init", "mixed"))])
    argv.extend(["--lm-budget", str(cfg.get("lm_budget", 5))])
    argv.extend(["--line-lm-budget", str(cfg.get("line_lm_budget", 3))])
    argv.extend(["--prompt-llm-fraction", "0"])
    argv.extend(["--archive-mode", str(cfg.get("archive_mode", "ultra_compact"))])
    argv.extend(["--immigrants", str(cfg.get("immigrants", 25))])
    argv.append("--db")
    argv.append("--runs-dir")

    seed_from_archive = cfg.get("seed_from_archive", 0)
    if seed_from_archive and int(seed_from_archive) > 0:
        argv.extend(["--seed-from-archive", str(int(seed_from_archive))])
    if cfg.get("arm"):
        argv.extend(["--arm", str(cfg["arm"])])
    if cfg.get("seed") is not None:
        argv.extend(["--seed", str(cfg["seed"])])

    return argv


def handler(job):
    cfg = job["input"]
    arm = cfg.get("arm", "unknown")
    seed = cfg.get("seed", "?")
    logger.info("Starting evolution: arm=%s seed=%s", arm, seed)

    for key in ("RAPBOT_DB_HOST", "RAPBOT_DB_PORT", "RAPBOT_DB_USER",
                "RAPBOT_DB_PASSWORD", "RAPBOT_DB_NAME", "RAPBOT_USE_DB"):
        if key.lower() in cfg:
            os.environ[key] = str(cfg.pop(key.lower()))

    sys.argv = _build_argv(cfg)

    os.chdir("/app")
    sys.path.insert(0, "/app")

    from scripts.run_verse_qd import main
    main()

    logger.info("Evolution completed: arm=%s seed=%s", arm, seed)
    return {"status": "completed", "arm": arm, "seed": seed}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
