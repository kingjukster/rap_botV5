"""
Optional JSONL operator events for mutation/crossover attribution.

Enable by calling ``set_operator_tracer`` from a runner when ``run_dir`` is set.
When a ``run_id`` is supplied, events are also mirrored to the database via
``evo_rhyme.db.insert_operator_event``.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evo_rhyme.repro import append_jsonl

logger = logging.getLogger(__name__)

_tracer_var: contextvars.ContextVar[Optional["OperatorTracer"]] = contextvars.ContextVar(
    "operator_tracer", default=None
)


@dataclass
class OperatorTracer:
    """Append one JSON object per line to ``operator_events.jsonl`` under run_dir.

    When ``run_id`` is set (> 0) events are also mirrored to the database.
    Callers should update ``gen`` each generation so the DB row has the right value.
    """

    run_dir: Path
    run_id: int = 0
    gen: int = 0

    def record(self, **kwargs: Any) -> None:
        row: Dict[str, Any] = dict(kwargs)
        row.setdefault("gen", self.gen)
        path = Path(self.run_dir) / "operator_events.jsonl"
        try:
            append_jsonl(path, row)
        except Exception as e:
            logger.warning("operator_events.jsonl append failed: %s", e)
        if self.run_id > 0:
            self._mirror_to_db(row)

    def _mirror_to_db(self, row: Dict[str, Any]) -> None:
        try:
            from evo_rhyme import db as _db
            if not _db.db_enabled():
                return
            operator_name = row.get("operator_name") or row.get("operator_kind", "unknown")
            _db.insert_operator_event(
                run_id=self.run_id,
                gen=row.get("gen", self.gen),
                operator=operator_name,
                candidate_id=row.get("candidate_id"),
                parents=row.get("parent_ids"),
                meta={k: v for k, v in row.items()
                      if k not in ("run_id", "gen", "operator_name", "operator_kind",
                                   "candidate_id", "parent_ids")},
            )
        except Exception as e:
            logger.debug("operator_events DB mirror failed: %s", e)


def set_operator_tracer(tracer: Optional[OperatorTracer]) -> contextvars.Token:
    return _tracer_var.set(tracer)


def reset_operator_tracer(token: contextvars.Token) -> None:
    _tracer_var.reset(token)


def get_operator_tracer() -> Optional[OperatorTracer]:
    return _tracer_var.get()


def record_operator_event(**kwargs: Any) -> None:
    """Record an event if a tracer is active.

    Accepted kwargs include ``operator_name``, ``operator_kind``, ``scope``,
    ``succeeded``, ``fallback``, ``candidate_id``, ``parent_ids``.
    """
    t = get_operator_tracer()
    if t is not None:
        t.record(**kwargs)
