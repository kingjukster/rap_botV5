"""REST API routes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from webapp.services.run_service import (
    list_runs as svc_list_runs,
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
)

router = APIRouter()


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
