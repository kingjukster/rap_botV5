"""Page routes (HTML)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import datetime as _dt
import decimal

from webapp.config import TEMPLATES_DIR
from webapp.services.run_service import (
    list_runs,
    get_run_summary,
    get_run_generations,
    get_run_candidates,
    get_run_top_candidates,
    get_run_archive,
    get_candidate,
    get_candidate_lineage,
    get_run_lineage,
    get_run_seeds,
    get_score_cache_recent,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
router = APIRouter()

def _make_json_safe(value: object) -> object:
    """Convert common non-JSON-serializable types (datetime/decimal) for Jinja2 `tojson`."""
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    return value


def _sanitize_dicts_for_json(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for item in items:
        safe = {}
        for k, v in item.items():
            safe[k] = _make_json_safe(v)
        out.append(safe)
    return out


def _dashboard_summary(runs: list) -> dict:
    """Compute summary stats from runs list: total and counts by status."""
    by_status = {"running": 0, "completed": 0, "failed": 0}
    for r in runs:
        s = (r.get("status") or "").lower()
        if s in by_status:
            by_status[s] += 1
    return {"total": len(runs), "by_status": by_status}


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, status: str | None = None):
    """Dashboard: list runs."""
    data = list_runs(limit=50, offset=0, status_filter=status)
    # Enrich each run with summary if we have it
    for r in data["runs"]:
        summary = get_run_summary(r["run_id"])
        if summary:
            r["last_best_fitness"] = summary.get("last_best_fitness")
            r["num_generations"] = summary.get("num_generations", 0)
        else:
            r["last_best_fitness"] = None
            r["num_generations"] = 0
    summary_stats = _dashboard_summary(data["runs"]) if data["runs"] else {"total": 0, "by_status": {"running": 0, "completed": 0, "failed": 0}}
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "runs": data["runs"],
            "total": data["total"],
            "status_filter": status,
            "summary": summary_stats,
        },
    )


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: int):
    """Run detail page."""
    run = get_run_summary(run_id)
    if not run:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/", status_code=302)
    generations = _sanitize_dicts_for_json(get_run_generations(run_id))
    candidates = _sanitize_dicts_for_json(get_run_candidates(run_id, gen=None)[:50])
    top_candidates = _sanitize_dicts_for_json(get_run_top_candidates(run_id, limit=12))
    has_archive = len(get_run_archive(run_id)) > 0
    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "run": run,
            "generations": generations,
            "candidates": candidates,
            "top_candidates": top_candidates,
            "has_archive": has_archive,
        },
    )


@router.get("/about", response_class=HTMLResponse)
def about(request: Request):
    """About / Help page."""
    return templates.TemplateResponse("about.html", {"request": request})


@router.get("/runs/{run_id}/generations", response_class=HTMLResponse)
def run_generations(request: Request, run_id: int):
    """Generations view: chart + table for a run."""
    run = get_run_summary(run_id)
    if not run:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/", status_code=302)
    generations = _sanitize_dicts_for_json(get_run_generations(run_id))
    return templates.TemplateResponse(
        "run_generations.html",
        {"request": request, "run": run, "generations": generations},
    )


@router.get("/runs/{run_id}/archive", response_class=HTMLResponse)
def archive_view(request: Request, run_id: int):
    """MAP-Elites archive explorer."""
    run = get_run_summary(run_id)
    if not run:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/", status_code=302)
    cells = get_run_archive(run_id)
    return templates.TemplateResponse(
        "archive.html",
        {"request": request, "run": run, "cells": cells},
    )


@router.get("/runs/{run_id}/lineage", response_class=HTMLResponse)
def run_lineage(request: Request, run_id: int, gen: int | None = None):
    """Lineage explorer for a run."""
    run = get_run_summary(run_id)
    if not run:
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/", status_code=302)
    edges = _sanitize_dicts_for_json(get_run_lineage(run_id, gen=gen, limit=2000, offset=0))
    return templates.TemplateResponse(
        "lineage.html",
        {"request": request, "run": run, "edges": edges, "gen_filter": gen},
    )


@router.get("/runs/{run_id}/candidates/{candidate_id}", response_class=HTMLResponse)
def candidate_detail(request: Request, run_id: int, candidate_id: int):
    """Candidate detail + lineage links."""
    run = get_run_summary(run_id)
    if not run:
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/", status_code=302)
    c = get_candidate(candidate_id)
    if not c or int(c.get("run_id") or -1) != int(run_id):
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url=f"/runs/{run_id}", status_code=302)

    lineage = _sanitize_dicts_for_json([get_candidate_lineage(candidate_id)])[0]
    return templates.TemplateResponse(
        "candidate_detail.html",
        {"request": request, "run": run, "candidate": _sanitize_dicts_for_json([c])[0], "lineage": lineage},
    )


@router.get("/runs/{run_id}/seeds", response_class=HTMLResponse)
def run_seeds(request: Request, run_id: int):
    """Seed bank view for a run."""
    run = get_run_summary(run_id)
    if not run:
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/", status_code=302)
    seeds = _sanitize_dicts_for_json(get_run_seeds(run_id, limit=200, offset=0))
    return templates.TemplateResponse(
        "seeds.html",
        {"request": request, "run": run, "seeds": seeds},
    )


@router.get("/score-cache", response_class=HTMLResponse)
def score_cache_recent(request: Request):
    """Score cache view (recent keys)."""
    rows = _sanitize_dicts_for_json(get_score_cache_recent(limit=200, offset=0))
    return templates.TemplateResponse(
        "score_cache.html",
        {"request": request, "rows": rows},
    )
