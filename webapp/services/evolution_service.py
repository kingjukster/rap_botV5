"""Evolution service: web-triggered evolution jobs via subprocess or Docker."""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from webapp.config import ROOT

logger = logging.getLogger(__name__)

# When running in web Docker container, we spawn the evolution container via Docker SDK.
# Set RAPBOT_DOCKER_NETWORK (and optionally RAPBOT_EVOLUTION_IMAGE, RAPBOT_HOST_PROJECT_PATH)
# to enable Docker-based evolution.


def _get_db():
    """Import evo_rhyme.db without loading the full evo_rhyme package (which requires numpy, torch, etc.)."""
    db_path = ROOT / "evo_rhyme" / "db.py"
    if not db_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("evo_rhyme.db", db_path)
    if spec is None or spec.loader is None:
        return None
    db = importlib.util.module_from_spec(spec)
    sys.modules["evo_rhyme.db"] = db
    spec.loader.exec_module(db)
    return db
_SONG_CATALOG_CACHE: Dict[str, Any] = {
    "path": None,
    "mtime_ns": None,
    "artists": [],
    "songs_by_artist": {},
}


def _get_script_path() -> Path:
    """Path to run_verse_qd.py."""
    return ROOT / "scripts" / "run_verse_qd.py"


def _can_use_docker() -> bool:
    """True if we should spawn evolution via Docker (web container has socket + config)."""
    if not os.path.exists("/var/run/docker.sock"):
        return False
    network = os.environ.get("RAPBOT_DOCKER_NETWORK", "").strip()
    return bool(network)


def _spawn_evolution_via_docker(
    run_id: int,
    cmd: List[str],
    env: Dict[str, str],
) -> bool:
    """Spawn evolution in a Docker container. Returns True if launched successfully."""
    try:
        import docker
    except ImportError:
        logger.warning("docker package not installed; cannot spawn evolution container")
        return False

    network = os.environ.get("RAPBOT_DOCKER_NETWORK", "").strip()
    image = os.environ.get("RAPBOT_EVOLUTION_IMAGE", "rap_botv5_evolution").strip()
    host_project = os.environ.get("RAPBOT_HOST_PROJECT_PATH", "").strip()

    # Build container command: ["python", "scripts/run_verse_qd.py", "--theme", ...]
    # cmd[0] is the python executable, cmd[1] is the script path - use relative path in container
    container_cmd = ["python", "scripts/run_verse_qd.py"] + cmd[2:]  # skip python, script path; keep --theme, --population, etc.

    env_list = [f"{k}={v}" for k, v in env.items()]

    volumes = None
    if host_project and os.path.isabs(host_project):
        volumes = {host_project: {"bind": "/app", "mode": "rw"}}
    elif not host_project:
        logger.warning(
            "RAPBOT_HOST_PROJECT_PATH not set; evolution container will use image code (may be stale)"
        )

    try:
        client = docker.from_env()
        client.containers.run(
            image,
            command=container_cmd,
            environment=env_list,
            network=network,
            volumes=volumes,
            detach=True,
            remove=True,
            working_dir="/app",
        )
        logger.info("Started evolution container run_id=%s", run_id)
        return True
    except docker.errors.ImageNotFound:
        logger.error(
            "Evolution image %s not found. Build with: docker compose build evolution",
            image,
        )
        return False
    except Exception as e:
        logger.exception("Failed to spawn evolution container: %s", e)
        return False


def _get_song_catalog_csv_path() -> Path:
    """Path to the song catalog CSV used for artist/song selectors."""
    configured = os.environ.get("RAPBOT_SONG_CATALOG_CSV", "").strip()
    if configured:
        p = Path(configured)
        return p if p.is_absolute() else (ROOT / p)
    return ROOT / "data" / "elite_songs_lines_clean.csv"


def _get_python_path() -> str:
    """Python executable (same as current process)."""
    return sys.executable


def _load_song_catalog() -> Tuple[List[str], Dict[str, List[Dict[str, str]]]]:
    """
    Load artist -> songs mapping from CSV, with mtime-based cache invalidation.
    Returns (artists, songs_by_artist).
    """
    csv_path = _get_song_catalog_csv_path()
    if not csv_path.exists():
        logger.warning("Song catalog CSV not found: %s", csv_path)
        return [], {}

    stat = csv_path.stat()
    if (
        _SONG_CATALOG_CACHE["path"] == str(csv_path)
        and _SONG_CATALOG_CACHE["mtime_ns"] == stat.st_mtime_ns
    ):
        return _SONG_CATALOG_CACHE["artists"], _SONG_CATALOG_CACHE["songs_by_artist"]

    songs_by_artist: Dict[str, Dict[str, Dict[str, str]]] = {}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                artist = str(row.get("artist") or "").strip()
                title = str(row.get("title") or "").strip()
                if not artist or not title:
                    continue
                song_id = str(row.get("song_id") or "").strip() or f"{artist}::{title}"
                artist_bucket = songs_by_artist.setdefault(artist, {})
                if song_id not in artist_bucket:
                    artist_bucket[song_id] = {"song_id": song_id, "title": title}
    except Exception as e:
        logger.exception("Failed to read song catalog CSV: %s", e)
        return [], {}

    artists = sorted(songs_by_artist.keys(), key=lambda x: x.lower())
    songs_map: Dict[str, List[Dict[str, str]]] = {}
    for artist in artists:
        songs = list(songs_by_artist[artist].values())
        songs.sort(key=lambda x: x["title"].lower())
        songs_map[artist] = songs

    _SONG_CATALOG_CACHE["path"] = str(csv_path)
    _SONG_CATALOG_CACHE["mtime_ns"] = stat.st_mtime_ns
    _SONG_CATALOG_CACHE["artists"] = artists
    _SONG_CATALOG_CACHE["songs_by_artist"] = songs_map
    return artists, songs_map


def list_song_artists() -> List[str]:
    """List artists that have at least one seedable song in the catalog.
    Prefers DB when enabled and populated; falls back to CSV."""
    db = _get_db()
    if db and db.db_enabled():
        artists = db.list_song_artists()
        if artists:
            return artists
    artists, _ = _load_song_catalog()
    return artists


def list_songs_for_artist(artist: str) -> List[Dict[str, str]]:
    """List songs for one artist from the catalog.
    Prefers DB when enabled and populated; falls back to CSV."""
    if not artist:
        return []
    db = _get_db()
    if db and db.db_enabled():
        songs = db.list_songs_for_artist(artist)
        if songs:
            return [{"song_id": s["song_id"], "title": s["title"]} for s in songs]
    _, songs_by_artist = _load_song_catalog()
    return list(songs_by_artist.get(artist, []))


def start_evolution(
    theme: str,
    population: int = 60,
    generations: int = 20,
    scheme: str = "AABB",
    init_mode: str = "mixed",
    seed_songs: Optional[List[Dict[str, str]]] = None,
) -> Optional[int]:
    """
    Start a QD evolution job in the background. Inserts run in DB, spawns subprocess,
    returns run_id immediately so the client can redirect to /runs/{run_id}.

    Returns run_id on success, None on failure (e.g. DB disabled or insert failed).
    """
    db = _get_db()
    if db is None:
        logger.warning("evo_rhyme.db not available; evolution requires DB")
        return None

    if not db.db_enabled():
        logger.warning("Database not enabled; evolution requires RAPBOT_USE_DB=1")
        return None

    theme_keywords = ",".join(w.strip() for w in theme.split(",") if w.strip()) or "general"
    config_json: Dict[str, Any] = {
        "source": "evolution_web",
        "theme": theme_keywords,
        "population": population,
        "generations": generations,
        "scheme": scheme,
        "init": init_mode,
    }
    if seed_songs:
        config_json["seed_songs"] = seed_songs

    run_id = db.insert_run(
        script_name="run_verse_qd",
        theme_keywords=theme_keywords,
        config_json=config_json,
    )
    if run_id is None or run_id <= 0:
        logger.warning("DB insert_run failed or returned invalid run_id")
        return None

    script = _get_script_path()
    if not script.exists():
        logger.error("Evolution script not found: %s", script)
        db.update_run_status(run_id, "failed", failure_reason="Script not found")
        return None

    cmd = [
        _get_python_path(),
        str(script),
        "--theme", theme_keywords,
        "--population", str(population),
        "--generations", str(generations),
        "--scheme", scheme,
        "--init", init_mode,
        "--db",
        "--run-id", str(run_id),
        "--runs-dir",
    ]
    if seed_songs:
        song_ids = [s.get("song_id") for s in seed_songs if s.get("song_id")]
        if song_ids:
            cmd.extend(["--seed-song-ids", ",".join(song_ids)])

    env = os.environ.copy()
    env["RAPBOT_USE_DB"] = "1"

    try:
        if _can_use_docker():
            ok = _spawn_evolution_via_docker(run_id, cmd, env)
            if not ok:
                db.update_run_status(
                    run_id, "failed",
                    failure_reason="Docker spawn failed (image not built? Run: docker compose build evolution)",
                )
                return None
            return run_id
        subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        logger.info("Started evolution subprocess run_id=%d", run_id)
    except Exception as e:
        logger.exception("Failed to spawn evolution: %s", e)
        db.update_run_status(
            run_id, "failed",
            failure_reason=f"Spawn failed: {type(e).__name__}: {e}",
        )
        return None

    return run_id
