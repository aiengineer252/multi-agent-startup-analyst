"""Drift detection: is the system quietly getting worse?

LLM pipelines rot without anyone touching the code. Providers silently update
models, you tweak a prompt, a spec changes, traffic shifts to different kinds
of ideas. None of that throws an exception — the reports keep looking fine
while the numbers underneath move.

So: freeze a BASELINE (a snapshot of metrics from a period you were happy
with), then compare a recent WINDOW against it.

Four kinds of drift are tracked, because they fail differently:

  QUALITY   judge score / spec compliance falling      -> answers getting worse
  BEHAVIOUR mean score or verdict mix shifting         -> the system changed its mind
  COST      tokens / dollars per run rising            -> silent bill growth
  LATENCY   p50 / p95 rising                           -> silent slowdown

A word on rigour: with a handful of runs these are directional signals, not
statistical tests. `confidence` is reported honestly as LOW below
`min_samples`, and you should treat a LOW-confidence DRIFT flag as
"go look", never as "it's proven".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .monitoring import summarize

DEFAULT_BASELINE_DIR = Path("evals/baselines")

OK, WARN, DRIFT, UNKNOWN = "OK", "WARN", "DRIFT", "UNKNOWN"


@dataclass
class Thresholds:
    """How much movement is tolerated before we complain.

    Defaults are deliberately loose — a noisy detector that cries wolf gets
    ignored, which is worse than no detector at all. Tighten them once you
    have enough history to know your real run-to-run variance.
    """

    min_samples: int = 5

    # Two-sided: score INFLATION is as suspicious as deflation (1-10 scale).
    score_delta_warn: float = 0.5
    score_delta_drift: float = 1.0

    # One-sided: only drops matter (0-1 scale).
    compliance_drop_warn: float = 0.05
    compliance_drop_drift: float = 0.10

    # One-sided: only drops matter (1-5 judge scale).
    judge_drop_warn: float = 0.30
    judge_drop_drift: float = 0.60

    # One-sided, relative: only increases matter.
    latency_increase_warn: float = 0.50     # +50%
    latency_increase_drift: float = 1.00    # +100%
    cost_increase_warn: float = 0.30
    cost_increase_drift: float = 0.60

    # Verdict mix, total-variation distance in 0-1.
    verdict_tvd_warn: float = 0.25
    verdict_tvd_drift: float = 0.40

    # Absolute error-rate increase.
    error_rate_warn: float = 0.05
    error_rate_drift: float = 0.15


@dataclass
class Signal:
    metric: str
    kind: str            # quality | behaviour | cost | latency | reliability
    status: str          # OK | WARN | DRIFT | UNKNOWN
    baseline: Optional[float]
    current: Optional[float]
    delta: Optional[float]
    note: str

    def __str__(self) -> str:
        icon = {OK: "✓", WARN: "!", DRIFT: "✗", UNKNOWN: "?"}[self.status]
        return f"{icon} {self.metric}: {self.note}"


@dataclass
class DriftReport:
    baseline_name: str
    baseline_runs: int
    current_runs: int
    confidence: str                     # HIGH | LOW
    signals: list[Signal] = field(default_factory=list)
    generated_at: str = ""

    @property
    def status(self) -> str:
        if any(s.status == DRIFT for s in self.signals):
            return DRIFT
        if any(s.status == WARN for s in self.signals):
            return WARN
        return OK

    @property
    def problems(self) -> list[Signal]:
        return [s for s in self.signals if s.status in (WARN, DRIFT)]

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------- #

def save_baseline(
    runs: list[dict], name: str = "default", directory: Path | str = DEFAULT_BASELINE_DIR
) -> Path:
    """Freeze the current metrics as the reference point.

    Baselines are committed to git on purpose — a code review that changes a
    prompt should show the baseline moving in the same commit, so the team can
    see the intended effect instead of discovering it in production.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_count": len(runs),
        "run_ids": [r["run_id"] for r in runs],
        "metrics": summarize(runs),
    }
    path = d / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_baseline(
    name: str = "default", directory: Path | str = DEFAULT_BASELINE_DIR
) -> Optional[dict]:
    path = Path(directory) / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_baselines(directory: Path | str = DEFAULT_BASELINE_DIR) -> list[str]:
    d = Path(directory)
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


# --------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------- #

def _two_sided(
    metric: str, kind: str, base: Optional[float], cur: Optional[float],
    warn: float, drift: float, unit: str = "",
) -> Signal:
    """Any movement in either direction is suspicious."""
    if base is None or cur is None:
        return Signal(metric, kind, UNKNOWN, base, cur, None, "no data on one side")
    delta = cur - base
    mag = abs(delta)
    status = DRIFT if mag >= drift else WARN if mag >= warn else OK
    return Signal(
        metric, kind, status, round(base, 3), round(cur, 3), round(delta, 3),
        f"{base:.2f}{unit} → {cur:.2f}{unit} ({delta:+.2f})",
    )


def _drop_only(
    metric: str, kind: str, base: Optional[float], cur: Optional[float],
    warn: float, drift: float,
) -> Signal:
    """Only a decrease is bad (compliance, judge quality)."""
    if base is None or cur is None:
        return Signal(metric, kind, UNKNOWN, base, cur, None, "no data on one side")
    delta = cur - base
    drop = -delta
    status = DRIFT if drop >= drift else WARN if drop >= warn else OK
    arrow = "improved" if delta > 0 else "dropped"
    return Signal(
        metric, kind, status, round(base, 3), round(cur, 3), round(delta, 3),
        f"{base:.3f} → {cur:.3f} ({arrow} {abs(delta):.3f})",
    )


def _increase_only(
    metric: str, kind: str, base: Optional[float], cur: Optional[float],
    warn: float, drift: float, unit: str = "",
) -> Signal:
    """Only a relative increase is bad (latency, cost)."""
    if base is None or cur is None or base == 0:
        return Signal(metric, kind, UNKNOWN, base, cur, None, "no data on one side")
    rel = (cur - base) / base
    status = DRIFT if rel >= drift else WARN if rel >= warn else OK
    return Signal(
        metric, kind, status, round(base, 4), round(cur, 4), round(rel, 3),
        f"{base:.2f}{unit} → {cur:.2f}{unit} ({rel:+.0%})",
    )


def _verdict_tvd(base: dict, cur: dict) -> tuple[Optional[float], str]:
    """Total variation distance between two verdict distributions.

    0.0 = identical mix, 1.0 = completely disjoint. This is the metric that
    catches "the system started saying NO-GO to everything" — a shift no
    single-run check would ever notice.
    """
    if not base or not cur:
        return None, "no verdict data"
    keys = set(base) | set(cur)
    bt, ct = sum(base.values()), sum(cur.values())
    if bt == 0 or ct == 0:
        return None, "no verdict data"
    tvd = 0.5 * sum(abs(base.get(k, 0) / bt - cur.get(k, 0) / ct) for k in keys)
    fmt = lambda d, t: ", ".join(  # noqa: E731
        f"{k} {v/t:.0%}" for k, v in sorted(d.items(), key=lambda x: -x[1])
    )
    return tvd, f"[{fmt(base, bt)}] → [{fmt(cur, ct)}]"


# --------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------- #

def compare(
    baseline: dict,
    current_runs: list[dict],
    thresholds: Optional[Thresholds] = None,
) -> DriftReport:
    """Compare a window of recent runs against a saved baseline."""
    t = thresholds or Thresholds()
    base = baseline.get("metrics", {})
    cur = summarize(current_runs)

    n = min(baseline.get("run_count", 0), len(current_runs))
    confidence = "HIGH" if n >= t.min_samples else "LOW"

    signals = [
        # --- BEHAVIOUR: has the system changed its mind? ---------------
        _two_sided(
            "mean_score", "behaviour",
            base.get("mean_score"), cur.get("mean_score"),
            t.score_delta_warn, t.score_delta_drift, "/10",
        ),
        # --- QUALITY ---------------------------------------------------
        _drop_only(
            "spec_compliance", "quality",
            base.get("mean_compliance"), cur.get("mean_compliance"),
            t.compliance_drop_warn, t.compliance_drop_drift,
        ),
        _drop_only(
            "judge_quality", "quality",
            base.get("judge_overall"), cur.get("judge_overall"),
            t.judge_drop_warn, t.judge_drop_drift,
        ),
        # --- LATENCY ---------------------------------------------------
        _increase_only(
            "latency_p50", "latency",
            base.get("latency_p50_s"), cur.get("latency_p50_s"),
            t.latency_increase_warn, t.latency_increase_drift, "s",
        ),
        _increase_only(
            "latency_p95", "latency",
            base.get("latency_p95_s"), cur.get("latency_p95_s"),
            t.latency_increase_warn, t.latency_increase_drift, "s",
        ),
        # --- COST ------------------------------------------------------
        _increase_only(
            "cost_per_run", "cost",
            base.get("cost_mean_usd"), cur.get("cost_mean_usd"),
            t.cost_increase_warn, t.cost_increase_drift, "$",
        ),
    ]

    # --- BEHAVIOUR: verdict mix ---------------------------------------
    tvd, note = _verdict_tvd(base.get("verdicts", {}), cur.get("verdicts", {}))
    if tvd is None:
        signals.append(
            Signal("verdict_mix", "behaviour", UNKNOWN, None, None, None, note)
        )
    else:
        status = (
            DRIFT if tvd >= t.verdict_tvd_drift
            else WARN if tvd >= t.verdict_tvd_warn
            else OK
        )
        signals.append(
            Signal("verdict_mix", "behaviour", status, None, round(tvd, 3),
                   round(tvd, 3), f"shift {tvd:.0%} — {note}")
        )

    # --- RELIABILITY: agent failure rate -------------------------------
    b_err = (base.get("error_runs", 0) / base["count"]) if base.get("count") else None
    c_err = (cur.get("error_runs", 0) / cur["count"]) if cur.get("count") else None
    if b_err is None or c_err is None:
        signals.append(
            Signal("error_rate", "reliability", UNKNOWN, b_err, c_err, None, "no data")
        )
    else:
        rise = c_err - b_err
        status = (
            DRIFT if rise >= t.error_rate_drift
            else WARN if rise >= t.error_rate_warn
            else OK
        )
        signals.append(
            Signal("error_rate", "reliability", status, round(b_err, 3),
                   round(c_err, 3), round(rise, 3),
                   f"{b_err:.0%} → {c_err:.0%} of runs had a failing agent")
        )

    return DriftReport(
        baseline_name=baseline.get("name", "unknown"),
        baseline_runs=baseline.get("run_count", 0),
        current_runs=len(current_runs),
        confidence=confidence,
        signals=signals,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
