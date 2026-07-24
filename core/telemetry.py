"""
Core Telemetry Module (core/telemetry.py)
Phase 10: Lightweight structured span logger for per-phase latency tracing.

Writes one JSONL line per span to eval_results/traces/trace_{run_id}.jsonl.
Used by the dashboard for latency waterfall views showing exactly where
time is spent: embedding → retrieval → generation → judge → attribution.

Usage:
    from core.telemetry import Tracer
    tracer = Tracer(run_id="20250724_123456")

    with tracer.span("retrieval", query_id="Q001") as span:
        chunks, sims, sources = retrieve(question, collection)
        span.set("chunks_retrieved", len(chunks))
        span.set("top_similarity", sims[0] if sims else 0)
"""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional


TRACE_DIR = os.path.join("eval_results", "traces")


class SpanContext:
    """Mutable context object for a single telemetry span."""

    def __init__(self, name: str, run_id: str, query_id: str):
        self.name = name
        self.run_id = run_id
        self.query_id = query_id
        self._start = time.perf_counter()
        self._extra: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Attach arbitrary metadata to this span (e.g. chunk count, token count)."""
        self._extra[key] = value

    def _duration_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "query_id": self.query_id,
            "phase": self.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": self._duration_ms(),
            **self._extra,
        }


class Tracer:
    """
    Per-run tracer that writes span events to a JSONL trace file.
    Thread-safe for sequential evaluation (one query at a time).
    """

    def __init__(self, run_id: str, enabled: bool = True):
        self.run_id = run_id
        self.enabled = enabled
        if enabled:
            os.makedirs(TRACE_DIR, exist_ok=True)
            self._trace_path = os.path.join(TRACE_DIR, f"trace_{run_id}.jsonl")
        else:
            self._trace_path = None

    @contextmanager
    def span(self, name: str, query_id: str = "") -> Generator[SpanContext, None, None]:
        """
        Context manager for a timed span.

        Example:
            with tracer.span("generation", query_id="Q001") as s:
                answer, p, c = generate_answer(...)
                s.set("prompt_tokens", p)
                s.set("completion_tokens", c)
        """
        ctx = SpanContext(name=name, run_id=self.run_id, query_id=query_id)
        try:
            yield ctx
        finally:
            if self.enabled and self._trace_path:
                try:
                    with open(self._trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(ctx.to_dict()) + "\n")
                except Exception:
                    pass  # Telemetry must never crash the eval pipeline

    def get_trace_path(self) -> Optional[str]:
        return self._trace_path


def load_trace(run_id: str) -> list:
    """Load all spans for a given run_id from its JSONL trace file."""
    path = os.path.join(TRACE_DIR, f"trace_{run_id}.jsonl")
    if not os.path.exists(path):
        return []
    spans = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    spans.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return spans


def list_trace_runs() -> list:
    """Return all run IDs that have trace files."""
    if not os.path.exists(TRACE_DIR):
        return []
    runs = []
    for fname in sorted(os.listdir(TRACE_DIR)):
        if fname.startswith("trace_") and fname.endswith(".jsonl"):
            run_id = fname[len("trace_"):-len(".jsonl")]
            runs.append(run_id)
    return runs
