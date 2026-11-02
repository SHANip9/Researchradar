"""
core/metrics.py — Application Metrics, Health Monitoring & Performance Tracking.

Provides a lightweight in-process metrics collector that records operation
latencies, success/failure counts, and exposes a health-check summary for
the dashboard and sidebar status indicators.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.config import APP_NAME, APP_VERSION
from core.logger import get_logger

log = get_logger("metrics")


# ─── Data Structures ──────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class RequestMetric:
    """Immutable record of a single tracked operation."""

    operation: str
    duration_ms: float
    success: bool
    timestamp: str
    metadata: dict = field(default_factory=dict)


# ─── Collector ────────────────────────────────────────────────
class MetricsCollector:
    """Collects, stores, and summarises application metrics.

    This is intentionally simple (no Prometheus, no StatsD) — a
    single-process Streamlit app doesn't need heavy telemetry.
    The ``get_health()`` dict can be displayed in the sidebar or
    exposed via a future ``/healthz`` endpoint.
    """

    _MAX_RECENT: int = 200

    def __init__(self) -> None:
        self._start_time: float = time.time()
        self._request_count: int = 0
        self._error_count: int = 0
        self._recent: list[RequestMetric] = []
        self._op_totals: dict[str, dict] = {}  # op → {count, total_ms, errors}

    # ── Recording ─────────────────────────────────────────────
    def record(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **metadata,
    ) -> None:
        """Records a single operation metric."""
        self._request_count += 1
        if not success:
            self._error_count += 1

        metric = RequestMetric(
            operation=operation,
            duration_ms=round(duration_ms, 2),
            success=success,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )
        self._recent.append(metric)
        if len(self._recent) > self._MAX_RECENT:
            self._recent = self._recent[-self._MAX_RECENT :]

        # Per-operation aggregates
        agg = self._op_totals.setdefault(
            operation, {"count": 0, "total_ms": 0.0, "errors": 0}
        )
        agg["count"] += 1
        agg["total_ms"] += duration_ms
        if not success:
            agg["errors"] += 1

        (log.info if success else log.warning)(
            "metric | op=%-24s | %7.1fms | ok=%s", operation, duration_ms, success
        )

    # ── Health Check ──────────────────────────────────────────
    def get_health(self) -> dict:
        """Returns a health-check summary dict."""
        uptime = time.time() - self._start_time
        error_rate = (self._error_count / max(self._request_count, 1)) * 100
        return {
            "status": "healthy" if error_rate < 10 else "degraded",
            "app": APP_NAME,
            "version": APP_VERSION,
            "uptime_seconds": round(uptime, 1),
            "total_requests": self._request_count,
            "error_count": self._error_count,
            "error_rate_pct": round(error_rate, 2),
        }

    # ── Summaries ─────────────────────────────────────────────
    def get_recent(self, n: int = 20) -> list[dict]:
        """Returns the *n* most recent metrics as plain dicts."""
        return [
            {
                "op": m.operation,
                "ms": m.duration_ms,
                "ok": m.success,
                "ts": m.timestamp,
            }
            for m in self._recent[-n:]
        ]

    def get_op_summary(self) -> dict[str, dict]:
        """Returns per-operation aggregate statistics."""
        summary = {}
        for op, agg in self._op_totals.items():
            count = agg["count"]
            summary[op] = {
                "count": count,
                "avg_ms": round(agg["total_ms"] / max(count, 1), 1),
                "errors": agg["errors"],
            }
        return summary


# ─── Global Singleton ─────────────────────────────────────────
metrics = MetricsCollector()

