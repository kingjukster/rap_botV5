"""
Rap Bot web application.

Run: uvicorn webapp.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.api.routes import router as api_router
from webapp.routes.pages import router as pages_router
from webapp.config import TEMPLATES_DIR, STATIC_DIR, ROOT
from webapp.services.run_service import check_db_connection

app = FastAPI(
    title="Rap Bot",
    description="Evolutionary rap verse generation dashboard",
    version="1.0",
)

# Ensure templates and static dirs exist
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(pages_router, tags=["pages"])
app.include_router(api_router, prefix="/api", tags=["api"])


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.get("/health/db", include_in_schema=False)
def health_db():
    """Verify database connectivity. Returns 503 if DB is unavailable."""
    if check_db_connection():
        return {"status": "ok", "db": "connected"}
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "db": "unavailable"},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return a simple error page for unhandled exceptions (500). Let HTTPException pass through."""
    if isinstance(exc, StarletteHTTPException):
        raise exc
    return templates.TemplateResponse(
        request,
        "error.html",
        {"request": request, "message": "An unexpected error occurred. Please try again or return to the dashboard."},
        status_code=500,
    )
