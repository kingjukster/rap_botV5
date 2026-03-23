"""Page routes (HTML)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import datetime as _dt
import decimal

from webapp.config import TEMPLATES_DIR
from webapp.services.run_service import (
    list_runs,
    get_run_counts_by_status,
    mark_stale_runs_failed,
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
    """Convert common non-JSON-serializable types (datetime/decimal) for Jinja2 `tojson`.
    Recursively handles nested dicts and lists."""
    if value is None:
        return None
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    return value


def _sanitize_dicts_for_json(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for item in items:
        safe = {}
        for k, v in item.items():
            safe[k] = _make_json_safe(v)
        out.append(safe)
    return out


def _sanitize_run_for_template(run: dict) -> dict:
    """Ensure run dict (including config_json) is JSON-safe for Jinja tojson."""
    out = {}
    for k, v in run.items():
        if k == "created_at":
            # Normalize to display string so template never calls .strftime on mixed types
            if v is None:
                out[k] = None
            elif hasattr(v, "strftime"):
                out[k] = v.strftime("%Y-%m-%d %H:%M")
            elif isinstance(v, str) and len(v) >= 16:
                out[k] = v[:16].replace("T", " ")
            else:
                out[k] = str(v) if v else None
        else:
            out[k] = _make_json_safe(v)
    return out


def _format_created_at(value: object) -> str | None:
    """Format created_at for display: datetime -> YYYY-MM-DD HH:MM, string -> truncated."""
    if value is None:
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, str) and len(value) >= 16:
        return value[:16].replace("T", " ")
    return str(value) if value else None


def _dashboard_summary(runs: list) -> dict:
    """Compute summary stats from runs list: total and counts by status."""
    by_status = {"running": 0, "completed": 0, "failed": 0}
    for r in runs:
        s = (r.get("status") or "").lower()
        if s in by_status:
            by_status[s] += 1
    return {"total": len(runs), "by_status": by_status}


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, status: str | None = None, stale_marked: int | None = None):
    """Dashboard: list runs."""
    data = list_runs(limit=50, offset=0, status_filter=status)
    # Enrich each run with summary if we have it, and sanitize for template
    for r in data["runs"]:
        summary = get_run_summary(r["run_id"])
        if summary:
            r["last_best_fitness"] = summary.get("last_best_fitness")
            r["num_generations"] = summary.get("num_generations", 0)
        else:
            r["last_best_fitness"] = None
            r["num_generations"] = 0
        r["created_at"] = _format_created_at(r.get("created_at"))
    summary_stats = get_run_counts_by_status()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "request": request,
            "runs": data["runs"],
            "total": data["total"],
            "status_filter": status,
            "summary": summary_stats,
            "stale_marked": stale_marked,
        },
    )


@router.post("/mark-stale-runs")
def mark_stale_runs_post(minutes: int = 30):
    """Mark runs that have been 'running' with no activity for minutes as failed. Redirect to dashboard."""
    n = mark_stale_runs_failed(minutes_idle=minutes)
    return RedirectResponse(url=f"/?stale_marked={n}", status_code=303)


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: int):
    """Run detail page."""
    run = get_run_summary(run_id)
    if not run:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/", status_code=302)
    run = _sanitize_run_for_template(run)
    generations = _sanitize_dicts_for_json(get_run_generations(run_id))
    candidates = _sanitize_dicts_for_json(get_run_candidates(run_id, gen=None)[:50])
    top_candidates = _sanitize_dicts_for_json(get_run_top_candidates(run_id, limit=12))
    has_archive = len(get_run_archive(run_id)) > 0
    return templates.TemplateResponse(
        request,
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
    return templates.TemplateResponse(request, "about.html", {"request": request})


@router.get("/runs/{run_id}/generations", response_class=HTMLResponse)
def run_generations(request: Request, run_id: int):
    """Generations view: chart + table for a run."""
    run = get_run_summary(run_id)
    if not run:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/", status_code=302)
    run = _sanitize_run_for_template(run)
    generations = _sanitize_dicts_for_json(get_run_generations(run_id))
    return templates.TemplateResponse(
        request,
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
    run = _sanitize_run_for_template(run)
    cells = _sanitize_dicts_for_json(get_run_archive(run_id))
    return templates.TemplateResponse(
        request,
        "archive.html",
        {"request": request, "run": run, "cells": cells},
    )


@router.get("/runs/{run_id}/lineage", response_class=HTMLResponse)
def run_lineage(request: Request, run_id: int, gen: str | None = None):
    """Lineage explorer for a run."""
    run = get_run_summary(run_id)
    if not run:
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/", status_code=302)
    run = _sanitize_run_for_template(run)
    gen_int: int | None = None
    if gen is not None and str(gen).strip():
        try:
            gen_int = int(gen)
        except (ValueError, TypeError):
            pass
    edges = _sanitize_dicts_for_json(get_run_lineage(run_id, gen=gen_int, limit=2000, offset=0))
    return templates.TemplateResponse(
        request,
        "lineage.html",
        {"request": request, "run": run, "edges": edges, "gen_filter": gen_int},
    )


@router.get("/runs/{run_id}/candidates/{candidate_id}", response_class=HTMLResponse)
def candidate_detail(request: Request, run_id: int, candidate_id: int):
    """Candidate detail + lineage links."""
    run = get_run_summary(run_id)
    if not run:
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url="/", status_code=302)
    run = _sanitize_run_for_template(run)
    c = get_candidate(candidate_id)
    if not c or int(c.get("run_id") or -1) != int(run_id):
        from fastapi.responses import RedirectResponse

        return RedirectResponse(url=f"/runs/{run_id}", status_code=302)

    lineage = _sanitize_dicts_for_json([get_candidate_lineage(candidate_id)])[0]
    return templates.TemplateResponse(
        request,
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
    run = _sanitize_run_for_template(run)
    seeds = _sanitize_dicts_for_json(get_run_seeds(run_id, limit=200, offset=0))
    return templates.TemplateResponse(
        request,
        "seeds.html",
        {"request": request, "run": run, "seeds": seeds},
    )


@router.get("/evolve", response_class=HTMLResponse)
def evolve_page(request: Request):
    """Evolve: start a verse evolution job from the web."""
    return templates.TemplateResponse(request, "evolve.html", {"request": request})


@router.get("/sql", response_class=HTMLResponse)
def sql_page(request: Request):
    """SQL query page."""
    return templates.TemplateResponse(request, "sql.html", {"request": request})


@router.get("/analysis", response_class=HTMLResponse)
def analysis_page(request: Request):
    """Evolution analysis dashboard."""
    return templates.TemplateResponse(request, "analysis.html", {"request": request})


@router.get("/score-cache", response_class=HTMLResponse)
def score_cache_recent(request: Request):
    """Score cache view (recent keys)."""
    rows = _sanitize_dicts_for_json(get_score_cache_recent(limit=200, offset=0))
    return templates.TemplateResponse(
        request,
        "score_cache.html",
        {"request": request, "rows": rows},
    )
