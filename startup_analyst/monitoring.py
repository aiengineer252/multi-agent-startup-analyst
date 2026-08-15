"""Observability: record every run as a structured row, then summarize.

Design choice: append-only JSONL at runs/runs.jsonl.

Why JSONL and not a database — one line per run, human-readable, greppable,
diffable, and appendable from concurrent processes without locking. You can
load a year of runs with pandas later if you outgrow it, and nothing here
needs a server to be running.

What gets recorded is deliberately *decision-relevant*: not raw logs, but the
numbers you would actually act on — cost, latency, contract compliance, the
scores each analyst gave, and the final verdict.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_RUN_LOG = Path("runs/runs.jsonl")

# Price per 1M tokens (input, output), USD.
#
# Only models with published prices we can state confidently are listed.
# Anything absent reports cost as None -> displayed as "n/a" rather than a
# made-up number. Add your own entries here for other providers; a wrong cost
# figure is worse than an honest blank.
PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-mythos-5": (10.00, 50.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "mock-1": (0.0, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """USD for one call, or None when we don't have authoritative pricing."""
    for known, (pin, pout) in PRICING.items():
        if known in model:
            return (input_tokens / 1_000_000) * pin + (
                output_tokens / 1_000_000
            ) * pout
    return None


# --------------------------------------------------------------------- #
# Record shapes
# --------------------------------------------------------------------- #

@dataclass
class AgentTelemetry:
    """Per-agent measurements for a single run."""

    key: str
    name: str
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None
    words: int = 0
    score: Optional[float] = None          # the analyst's own 1-10 score
    compliance: float = 1.0                # 0-1, from spec.compliance_score()
    violations: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class RunRecord:
    """One end-to-end analysis. This is the unit drift detection compares."""

    run_id: str
    timestamp: str
    idea_preview: str
    idea_hash: str
    provider: str
    model: str
    web_search: bool
    total_latency_s: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: Optional[float]
    verdict: Optional[str]
    mean_score: Optional[float]
    mean_compliance: float
    error_count: int
    agents: list[dict]
    tags: list[str] = field(default_factory=list)
    judge: Optional[dict] = None           # filled in by evals.py when judged
    host: str = field(default_factory=platform.node)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------- #

def append_run(record: RunRecord, path: Path | str = DEFAULT_RUN_LOG) -> Path:
    """Append one run. Creates the file/dir on first use.

    Opened in append mode per-write so parallel processes interleave safely
    (POSIX appends under the typical line size are atomic; on Windows this is
    still fine for the single-writer case this tool has).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(record.to_json() + "\n")
    return p


def load_runs(
    path: Path | str = DEFAULT_RUN_LOG, limit: Optional[int] = None
) -> list[dict]:
    """Read runs oldest-first. Corrupt lines are skipped, not fatal —
    a half-written line from a killed process should not break your dashboard.
    """
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:] if limit else rows


# --------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------- #

def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def _pct(values: list[float], q: float) -> Optional[float]:
    """Simple nearest-rank percentile — no numpy dependency."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, round(q * (len(vals) - 1))))
    return vals[idx]


def summarize(runs: list[dict]) -> dict:
    """Roll a list of run records into the numbers worth looking at."""
    if not runs:
        return {"count": 0}

    latencies = [r["total_latency_s"] for r in runs]
    costs = [r["total_cost_usd"] for r in runs if r.get("total_cost_usd") is not None]

    verdicts: dict[str, int] = {}
    for r in runs:
        v = r.get("verdict") or "UNPARSED"
        verdicts[v] = verdicts.get(v, 0) + 1

    # Per-agent rollup so you can see WHICH analyst is slow/expensive/sloppy.
    per_agent: dict[str, dict] = {}
    for r in runs:
        for a in r.get("agents", []):
            slot = per_agent.setdefault(
                a["key"],
                {"name": a["name"], "lat": [], "cost": [], "comp": [], "score": [],
                 "violations": 0, "errors": 0},
            )
            slot["lat"].append(a.get("latency_s", 0.0))
            if a.get("cost_usd") is not None:
                slot["cost"].append(a["cost_usd"])
            slot["comp"].append(a.get("compliance", 1.0))
            if a.get("score") is not None:
                slot["score"].append(a["score"])
            slot["violations"] += len(a.get("violations", []))
            if a.get("error"):
                slot["errors"] += 1

    agents_summary = {
        k: {
            "name": v["name"],
            "mean_latency_s": _mean(v["lat"]),
            "mean_cost_usd": _mean(v["cost"]) if v["cost"] else None,
            "mean_compliance": _mean(v["comp"]),
            "mean_score": _mean(v["score"]),
            "total_violations": v["violations"],
            "errors": v["errors"],
        }
        for k, v in per_agent.items()
    }

    judged = [r["judge"]["overall"] for r in runs if r.get("judge")]

    return {
        "count": len(runs),
        "first": runs[0]["timestamp"],
        "last": runs[-1]["timestamp"],
        "providers": sorted({r["provider"] for r in runs}),
        "models": sorted({r["model"] for r in runs}),
        "latency_mean_s": _mean(latencies),
        "latency_p50_s": _pct(latencies, 0.50),
        "latency_p95_s": _pct(latencies, 0.95),
        "cost_mean_usd": _mean(costs) if costs else None,
        "cost_total_usd": sum(costs) if costs else None,
        "mean_score": _mean([r.get("mean_score") for r in runs]),
        "mean_compliance": _mean([r.get("mean_compliance", 1.0) for r in runs]),
        "judge_overall": _mean(judged) if judged else None,
        "error_runs": sum(1 for r in runs if r.get("error_count", 0) > 0),
        "verdicts": verdicts,
        "agents": agents_summary,
    }


def filter_runs(
    runs: list[dict],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    tag: Optional[str] = None,
    idea_hash: Optional[str] = None,
) -> list[dict]:
    out = runs
    if provider:
        out = [r for r in out if r.get("provider") == provider]
    if model:
        out = [r for r in out if r.get("model") == model]
    if tag:
        out = [r for r in out if tag in r.get("tags", [])]
    if idea_hash:
        out = [r for r in out if r.get("idea_hash") == idea_hash]
    return out


def resolve_log_path(explicit: Optional[str] = None) -> Path:
    """CLI flag > env var > default. Lets CI point at a scratch file."""
    return Path(explicit or os.environ.get("ANALYST_RUN_LOG") or DEFAULT_RUN_LOG)
