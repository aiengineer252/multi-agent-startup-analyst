"""Evaluation: three layers, cheapest first.

    Layer 1  CONTRACT   spec.check()      free, instant, deterministic
    Layer 2  BEHAVIOUR  golden cases      one pipeline run per case
    Layer 3  QUALITY    LLM-as-judge      one extra model call per report

Run them in that order and stop early when a cheap layer fails — there is no
point paying a judge model to rate a report that was truncated halfway.

Layer 1 catches *format* regressions (missing sections, no score, refusals).
Layer 2 catches *reasoning* regressions (calling an obvious dud a STRONG GO).
Layer 3 catches *quality* drift that still passes both (vaguer, more generic,
hedgier answers that are technically well-formed).
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from .golden_cases import GOLDEN_CASES, GoldenCase
from .orchestrator import RunResult, StartupAnalyst
from .providers import LLMProvider
from .specs import SPECS, OutputSpec


# --------------------------------------------------------------------- #
# Layer 3: LLM-as-judge
# --------------------------------------------------------------------- #

JUDGE_SYSTEM = (
    "You are a strict evaluator of analyst reports. You grade the QUALITY of "
    "analysis, not whether you agree with its conclusion. You are hard to "
    "impress: a 5 means genuinely excellent and rare, a 3 means competent but "
    "unremarkable, a 1 means it failed at this dimension. Generic advice that "
    "would apply to any startup is never above a 2 on specificity. "
    "You reply with JSON only — no prose, no code fences."
)


def _judge_prompt(idea: str, report: str, spec: OutputSpec) -> str:
    criteria = "\n".join(
        f'- "{c.id}": {c.question}' for c in spec.rubric
    )
    keys = ", ".join(f'"{c.id}"' for c in spec.rubric)
    return (
        f"Startup idea being analyzed:\n<idea>\n{idea}\n</idea>\n\n"
        f"Analyst report to grade:\n<report>\n{report}\n</report>\n\n"
        f"Grade the report on each criterion, 1-5:\n{criteria}\n\n"
        f"Reply with JSON exactly in this shape, using the keys {keys}:\n"
        '{"<criterion_id>": {"score": <1-5>, "why": "<one short sentence>"}}'
    )


def _parse_judge_json(raw: str) -> Optional[dict]:
    """Models sometimes wrap JSON in prose or code fences. Dig it out."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)   # first {...} block
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def judge_report(
    idea: str, report: str, spec: OutputSpec, judge: LLMProvider
) -> Optional[dict]:
    """Score one report against its rubric. Returns per-criterion scores plus
    a weighted `overall` on the same 1-5 scale, or None if judging failed."""
    if not spec.rubric:
        return None

    completion = judge.complete(JUDGE_SYSTEM, _judge_prompt(idea, report, spec))
    parsed = _parse_judge_json(completion.text)
    if not parsed:
        return None

    scores: dict[str, float] = {}
    notes: dict[str, str] = {}
    for c in spec.rubric:
        entry = parsed.get(c.id)
        if isinstance(entry, dict) and "score" in entry:
            try:
                scores[c.id] = max(1.0, min(5.0, float(entry["score"])))
                notes[c.id] = str(entry.get("why", ""))[:200]
            except (TypeError, ValueError):
                continue
        elif isinstance(entry, (int, float)):
            scores[c.id] = max(1.0, min(5.0, float(entry)))

    if not scores:
        return None

    weights = {c.id: c.weight for c in spec.rubric}
    total_w = sum(weights[k] for k in scores)
    overall = sum(scores[k] * weights[k] for k in scores) / total_w

    return {
        "overall": round(overall, 2),
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "notes": notes,
        "judge_model": judge.model,
    }


def judge_run(
    idea: str, run: RunResult, judge: LLMProvider, agent_keys: Optional[list] = None
) -> dict:
    """Judge every (or selected) report in a completed run and attach the
    result to its RunRecord so monitoring and drift can track quality too."""
    per_agent: dict[str, dict] = {}
    for res in run.results:
        if agent_keys and res.key not in agent_keys:
            continue
        verdict = judge_report(idea, res.report, SPECS[res.key], judge)
        if verdict:
            per_agent[res.key] = verdict

    overall = (
        round(statistics.fmean(v["overall"] for v in per_agent.values()), 2)
        if per_agent
        else None
    )
    payload = {"overall": overall, "per_agent": per_agent, "judge_model": judge.model}
    run.record.judge = payload
    return payload


# --------------------------------------------------------------------- #
# Layer 2: golden-case assertions
# --------------------------------------------------------------------- #

@dataclass
class Assertion:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'}  {self.name}: {self.detail}"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    assertions: list[Assertion] = field(default_factory=list)
    verdict: Optional[str] = None
    mean_score: Optional[float] = None
    mean_compliance: float = 0.0
    latency_s: float = 0.0
    cost_usd: Optional[float] = None
    judge_overall: Optional[float] = None

    @property
    def failures(self) -> list[Assertion]:
        return [a for a in self.assertions if not a.passed]


def check_case(case: GoldenCase, run: RunResult) -> CaseResult:
    """Apply a golden case's expectations to a completed run."""
    rec = run.record
    a: list[Assertion] = []

    # Every agent must have actually produced something.
    a.append(
        Assertion(
            "no_agent_errors",
            rec.error_count == 0,
            f"{rec.error_count} agent(s) errored",
        )
    )

    # Contract compliance floor.
    a.append(
        Assertion(
            "spec_compliance",
            rec.mean_compliance >= case.min_compliance,
            f"mean compliance {rec.mean_compliance:.2f} "
            f"(floor {case.min_compliance})",
        )
    )

    if case.expect_verdict_in:
        a.append(
            Assertion(
                "verdict_in_expected",
                rec.verdict in case.expect_verdict_in,
                f"got {rec.verdict!r}, expected one of "
                f"{list(case.expect_verdict_in)}",
            )
        )

    if case.expect_mean_score_max is not None:
        ok = rec.mean_score is not None and rec.mean_score <= case.expect_mean_score_max
        a.append(
            Assertion(
                "mean_score_ceiling",
                ok,
                f"mean score {rec.mean_score} <= {case.expect_mean_score_max}",
            )
        )

    if case.expect_mean_score_min is not None:
        ok = rec.mean_score is not None and rec.mean_score >= case.expect_mean_score_min
        a.append(
            Assertion(
                "mean_score_floor",
                ok,
                f"mean score {rec.mean_score} >= {case.expect_mean_score_min}",
            )
        )

    by_key = {r.key: r for r in run.results}
    for agent_key, keywords in case.expect_mentions.items():
        report = by_key[agent_key].report.lower() if agent_key in by_key else ""
        hit = next((k for k in keywords if k.lower() in report), None)
        a.append(
            Assertion(
                f"mentions[{agent_key}]",
                hit is not None,
                f"found {hit!r}" if hit else f"none of {list(keywords)} present",
            )
        )

    return CaseResult(
        case_id=case.id,
        passed=all(x.passed for x in a),
        assertions=a,
        verdict=rec.verdict,
        mean_score=rec.mean_score,
        mean_compliance=rec.mean_compliance,
        latency_s=rec.total_latency_s,
        cost_usd=rec.total_cost_usd,
        judge_overall=(rec.judge or {}).get("overall"),
    )


# --------------------------------------------------------------------- #
# Suite runner
# --------------------------------------------------------------------- #

@dataclass
class SuiteResult:
    cases: list[CaseResult]
    provider: str
    model: str

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.cases)

    @property
    def pass_rate(self) -> float:
        return sum(c.passed for c in self.cases) / len(self.cases) if self.cases else 0.0

    def summary(self) -> dict:
        judged = [c.judge_overall for c in self.cases if c.judge_overall is not None]
        costs = [c.cost_usd for c in self.cases if c.cost_usd is not None]
        return {
            "provider": self.provider,
            "model": self.model,
            "cases": len(self.cases),
            "passed": sum(c.passed for c in self.cases),
            "pass_rate": round(self.pass_rate, 3),
            "mean_compliance": round(
                statistics.fmean([c.mean_compliance for c in self.cases]), 3
            )
            if self.cases
            else 0.0,
            "judge_overall": round(statistics.fmean(judged), 2) if judged else None,
            "total_cost_usd": round(sum(costs), 4) if costs else None,
            "failures": {
                c.case_id: [str(f) for f in c.failures]
                for c in self.cases
                if not c.passed
            },
        }


def run_suite(
    provider: LLMProvider,
    cases: tuple[GoldenCase, ...] = GOLDEN_CASES,
    judge: Optional[LLMProvider] = None,
    on_status: Optional[Callable[[str], None]] = None,
    log_path: Optional[str] = None,
) -> SuiteResult:
    """Run every golden case end-to-end and grade it.

    Eval runs are tagged 'eval' in the run log so they can be excluded from
    production dashboards — otherwise your deliberately-terrible test ideas
    would drag down the real quality metrics.
    """

    def notify(msg: str) -> None:
        if on_status:
            on_status(msg)

    analyst = StartupAnalyst(provider, log_path=log_path, tags=["eval"])
    results: list[CaseResult] = []

    for case in cases:
        notify(f"▶ {case.id} — running pipeline...")
        run = analyst.analyze(case.idea, persist=True)

        if judge:
            notify(f"  ⚖ {case.id} — judging quality...")
            judge_run(case.idea, run, judge)

        outcome = check_case(case, run)
        results.append(outcome)
        notify(
            f"{'✓' if outcome.passed else '✗'} {case.id} — "
            f"{sum(a.passed for a in outcome.assertions)}/"
            f"{len(outcome.assertions)} assertions"
        )

    return SuiteResult(
        cases=results, provider=provider.name, model=provider.model
    )


def as_dict(suite: SuiteResult) -> dict:
    return {"summary": suite.summary(), "cases": [asdict(c) for c in suite.cases]}
