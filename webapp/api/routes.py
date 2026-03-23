"""REST API routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse

from webapp.services.evolution_service import (
    start_evolution,
    list_song_artists,
    list_songs_for_artist,
)
from webapp.services.analysis_service import get_analysis_data
from webapp.services.run_service import (
    list_runs as svc_list_runs,
    mark_stale_runs_failed,
    get_run,
    get_run_generations,
    get_run_candidates,
    get_run_archive,
    get_run_top_candidates,
    get_candidate,
    get_run_lineage,
    get_candidate_lineage,
    get_run_seeds,
    get_score_cache_recent,
    list_experiments,
    get_experiment,
    list_experiment_arms,
    list_experiment_runs,
    get_control_impact_report,
    execute_sql,
)

router = APIRouter()


@router.get("/analysis")
def api_analysis():
    """Evolution analysis: run stats, fitness trend, config dominance, stagnation."""
    return get_analysis_data()


@router.post("/sql")
def api_execute_sql(
    sql: str = Body(..., embed=True),
    max_rows: int = Body(1000, embed=True, ge=1, le=5000),
):
    """Execute a read-only SQL query. Returns columns, rows, and row_count or error."""
    result = execute_sql(sql=sql, max_rows=max_rows)
    if "error" in result:
        status = 503 if "Database is not available" in result["error"] else 400
        return JSONResponse(status_code=status, content=result)
    return result


@router.post("/runs/mark-stale")
def api_mark_stale_runs(
    minutes: int = Query(30, ge=5, le=1440, description="Mark runs stale if no activity for this many minutes"),
):
    """Mark runs as failed if they have been 'running' with no activity for the given minutes."""
    n = mark_stale_runs_failed(minutes_idle=minutes)
    return {"marked": n, "message": f"Marked {n} stale run(s) as failed"}


@router.get("/runs", response_model=Dict[str, Any])
def api_list_runs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="Filter by status: running, completed, failed"),
):
    """List runs with pagination."""
    return svc_list_runs(limit=limit, offset=offset, status_filter=status)


@router.get("/runs/{run_id}")
def api_get_run(run_id: int):
    """Get run detail."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/runs/{run_id}/generations", response_model=List[Dict[str, Any]])
def api_get_generations(run_id: int):
    """Get generation history for a run."""
    gens = get_run_generations(run_id)
    return gens


@router.get("/runs/{run_id}/candidates", response_model=List[Dict[str, Any]])
def api_get_candidates(
    run_id: int,
    gen: Optional[int] = Query(None, description="Filter by generation"),
):
    """Get candidates for a run."""
    return get_run_candidates(run_id, gen=gen)


@router.get("/runs/{run_id}/top-candidates", response_model=List[Dict[str, Any]])
def api_get_top_candidates(
    run_id: int,
    limit: int = Query(20, ge=1, le=200),
    candidate_type: Optional[str] = Query(None),
    scheme: Optional[str] = Query(None),
    gen: Optional[int] = Query(None),
):
    """Get top candidates by fitness for a run."""
    return get_run_top_candidates(run_id, limit=limit, candidate_type=candidate_type, scheme=scheme, gen=gen)


@router.get("/runs/{run_id}/archive", response_model=List[Dict[str, Any]])
def api_get_archive(run_id: int):
    """Get MAP-Elites archive cells (QD runs only)."""
    return get_run_archive(run_id)


@router.get("/candidates/{candidate_id}", response_model=Dict[str, Any])
def api_get_candidate(candidate_id: int):
    """Get candidate detail."""
    c = get_candidate(candidate_id)
    if not c:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return c


@router.get("/candidates/{candidate_id}/lineage", response_model=Dict[str, Any])
def api_get_candidate_lineage(candidate_id: int):
    """Get immediate lineage edges for a candidate."""
    return get_candidate_lineage(candidate_id)


@router.get("/runs/{run_id}/lineage", response_model=List[Dict[str, Any]])
def api_get_run_lineage(
    run_id: int,
    gen: Optional[int] = Query(None, description="Filter by generation"),
    limit: int = Query(2000, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    """Get lineage edges for a run."""
    return get_run_lineage(run_id, gen=gen, limit=limit, offset=offset)


@router.get("/runs/{run_id}/seeds", response_model=List[Dict[str, Any]])
def api_get_run_seeds(
    run_id: int,
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    """Get seed bank entries for a run."""
    return get_run_seeds(run_id, limit=limit, offset=offset)


@router.get("/score-cache/recent", response_model=List[Dict[str, Any]])
def api_get_score_cache_recent(
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    """List recent score_cache keys."""
    return get_score_cache_recent(limit=limit, offset=offset)


@router.get("/experiments", response_model=List[Dict[str, Any]])
def api_list_experiments(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List control experiments."""
    return list_experiments(limit=limit, offset=offset)


@router.get("/experiments/{experiment_id}")
def api_get_experiment(experiment_id: int):
    """Get experiment detail."""
    exp = get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp


@router.get("/experiments/{experiment_id}/arms", response_model=List[Dict[str, Any]])
def api_list_experiment_arms(experiment_id: int):
    """List arms for an experiment."""
    return list_experiment_arms(experiment_id)


@router.get("/experiments/{experiment_id}/runs", response_model=List[Dict[str, Any]])
def api_list_experiment_runs(
    experiment_id: int,
    arm_id: Optional[int] = Query(None),
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List runs for an experiment, optionally filtered by arm."""
    return list_experiment_runs(experiment_id, arm_id=arm_id, limit=limit, offset=offset)


@router.get("/experiments/{experiment_id}/impact")
def api_get_experiment_impact(
    experiment_id: int,
    arm_id: Optional[int] = Query(None),
    aggregation: str = Query("best", description="best | top_k_mean | mean"),
    top_k: int = Query(5, ge=1, le=20),
):
    """Get control impact report for an experiment (mean diffs, CI, correlations)."""
    report = get_control_impact_report(
        experiment_id, arm_id=arm_id, aggregation_mode=aggregation, top_k=top_k
    )
    if report is None:
        raise HTTPException(status_code=503, detail="Report not available")
    return report


@router.post("/evolution/start")
def api_evolution_start(
    theme: str = Body(..., embed=True),
    population: int = Body(60, embed=True, ge=10, le=200),
    generations: int = Body(20, embed=True, ge=3, le=100),
    scheme: str = Body("AABB", embed=True),
    init_mode: str = Body("mixed", embed=True),
    seed_songs: Optional[List[Dict[str, str]]] = Body(None, embed=True),
):
    """Start a QD evolution job. Returns run_id for redirect to /runs/{run_id}.
    seed_songs: optional list of {song_id, artist, title} to seed initial population."""
    run_id = start_evolution(
        theme=theme,
        population=population,
        generations=generations,
        scheme=scheme,
        init_mode=init_mode,
        seed_songs=seed_songs,
    )
    if run_id is None:
        raise HTTPException(
            status_code=503,
            detail="Evolution requires database (RAPBOT_USE_DB=1). Could not start job.",
        )
    return {"run_id": run_id, "message": f"Evolution started. Redirect to /runs/{run_id} to monitor."}


@router.get("/evolution/artists", response_model=List[str])
def api_evolution_artists():
    """List artists available for song-seeded evolution."""
    return list_song_artists()


@router.get("/evolution/songs", response_model=List[Dict[str, str]])
def api_evolution_songs(
    artist: str = Query(..., min_length=1, description="Artist name"),
):
    """List songs for one artist available for song-seeded evolution."""
    return list_songs_for_artist(artist=artist)


@router.get("/runs/{run_id}/progress")
def api_get_run_progress(run_id: int) -> Dict[str, Any]:
    """Get run progress: status, current generation index, and total generations (if in config)."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    gens = get_run_generations(run_id)
    config = run.get("config_json") or {}
    total = config.get("generations") if isinstance(config, dict) else None
    return {
        "run_id": run_id,
        "status": run.get("status", "unknown"),
        "current_generation": len(gens),
        "total_generations": total,
    }
