"""Core orchestration — provider-agnostic, self-measuring.

Flow:
    1. The five SPECIALISTS run in PARALLEL (one thread each) against the
       same startup idea. Each gets its persona (agents.py) plus its output
       contract (specs.py).
    2. Every report is immediately checked against its spec — free, instant,
       no API call — so a truncated or off-contract answer is caught here
       rather than discovered by a human three days later.
    3. The five reports are bundled and handed to the SYNTHESIZER, which
       writes the memo and picks one of the four allowed verdicts.
    4. Everything measured along the way is packed into a RunRecord and
       appended to runs/runs.jsonl.

A failed agent does NOT abort the run: its error is captured, the section is
marked, and the other four still produce a memo. Partial output beats no
output, and the run log records exactly what degraded.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .agents import SPECIALISTS, SYNTHESIZER, build_prompt
from .monitoring import (
    AgentTelemetry,
    RunRecord,
    append_run,
    estimate_cost,
    new_run_id,
    utc_now,
)
from .providers import LLMProvider
from .specs import SPECS


@dataclass
class AgentResult:
    key: str
    name: str
    emoji: str
    report: str
    telemetry: Optional[AgentTelemetry] = None


@dataclass
class RunResult:
    """What analyze() returns: the reports plus the measurement record."""

    results: list[AgentResult]
    record: RunRecord

    @property
    def memo(self) -> AgentResult:
        return self.results[-1]

    @property
    def specialists(self) -> list[AgentResult]:
        return self.results[:-1]


class StartupAnalyst:
    def __init__(
        self,
        provider: LLMProvider,
        log_path: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ):
        self.provider = provider
        self.log_path = log_path
        self.tags = tags or []

    # ------------------------------------------------------------------ #
    # One agent: call the model, then grade the answer against its spec
    # ------------------------------------------------------------------ #

    def _run_agent(self, agent: dict, prompt: str, allow_web: bool) -> AgentResult:
        spec = SPECS[agent["key"]]
        tel = AgentTelemetry(key=agent["key"], name=agent["name"])

        try:
            completion = self.provider.complete(agent["system"], prompt, allow_web)
            text = completion.text
            tel.latency_s = round(completion.latency_s, 2)
            tel.input_tokens = completion.input_tokens
            tel.output_tokens = completion.output_tokens
            tel.cost_usd = estimate_cost(
                completion.model, completion.input_tokens, completion.output_tokens
            )
        except Exception as exc:  # network, auth, rate limit, bad model name
            tel.error = f"{type(exc).__name__}: {exc}"
            text = f"_Agent failed: {tel.error}_"

        # Grade against the contract — the same spec object that wrote the prompt.
        tel.words = len(text.split())
        tel.score = spec.extract_score(text)
        tel.compliance = round(spec.compliance_score(text), 3)
        tel.violations = [str(x) for x in spec.check(text)]

        return AgentResult(
            key=agent["key"],
            name=agent["name"],
            emoji=agent["emoji"],
            report=text,
            telemetry=tel,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        idea: str,
        on_status: Optional[Callable[[str], None]] = None,
        persist: bool = True,
    ) -> RunResult:
        """Run the full pipeline and return reports + telemetry."""

        def notify(msg: str) -> None:
            if on_status:
                on_status(msg)

        started = time.time()
        results: dict[str, AgentResult] = {}

        # ---- Stage 1: specialists in parallel -------------------------- #
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(SPECIALISTS)
        ) as pool:
            futures = {}
            for agent in SPECIALISTS:
                notify(f"{agent['emoji']} {agent['name']} — analyzing...")
                prompt = build_prompt(agent, idea=idea)
                futures[pool.submit(self._run_agent, agent, prompt, True)] = agent

            for future in concurrent.futures.as_completed(futures):
                agent = futures[future]
                res = future.result()
                results[agent["key"]] = res
                tel = res.telemetry
                if tel and tel.error:
                    notify(f"{agent['emoji']} {agent['name']} — FAILED ✗")
                else:
                    flag = "" if tel.compliance >= 0.9 else "  ⚠ off-contract"
                    notify(
                        f"{agent['emoji']} {agent['name']} — done ✓ "
                        f"({tel.latency_s}s){flag}"
                    )

        ordered = [results[a["key"]] for a in SPECIALISTS]

        # ---- Stage 2: synthesis ---------------------------------------- #
        notify(f"{SYNTHESIZER['emoji']} {SYNTHESIZER['name']} — writing memo...")
        briefing = "\n\n---\n\n".join(f"## {r.name}\n\n{r.report}" for r in ordered)
        memo_prompt = build_prompt(SYNTHESIZER, idea=idea, reports=briefing)
        memo = self._run_agent(SYNTHESIZER, memo_prompt, False)
        notify(f"{SYNTHESIZER['emoji']} {SYNTHESIZER['name']} — done ✓")
        ordered.append(memo)

        # ---- Assemble the run record ----------------------------------- #
        record = self._build_record(idea, ordered, time.time() - started)
        if persist:
            append_run(record, self.log_path or "runs/runs.jsonl")

        return RunResult(results=ordered, record=record)

    def _build_record(
        self, idea: str, results: list[AgentResult], elapsed: float
    ) -> RunRecord:
        tels = [r.telemetry for r in results if r.telemetry]
        costs = [t.cost_usd for t in tels if t.cost_usd is not None]
        spec_scores = [
            t.score for t in tels if t.score is not None and t.key != "memo"
        ]
        memo_spec = SPECS["memo"]

        return RunRecord(
            run_id=new_run_id(),
            timestamp=utc_now(),
            idea_preview=idea.strip().replace("\n", " ")[:160],
            idea_hash=hashlib.sha256(idea.strip().encode()).hexdigest()[:12],
            provider=self.provider.name,
            model=self.provider.model,
            web_search=getattr(self.provider, "use_web_search", False),
            total_latency_s=round(elapsed, 2),
            total_input_tokens=sum(t.input_tokens for t in tels),
            total_output_tokens=sum(t.output_tokens for t in tels),
            total_cost_usd=round(sum(costs), 4) if costs else None,
            verdict=memo_spec.extract_verdict(results[-1].report),
            mean_score=(
                round(sum(spec_scores) / len(spec_scores), 2) if spec_scores else None
            ),
            mean_compliance=round(
                sum(t.compliance for t in tels) / len(tels), 3
            ) if tels else 0.0,
            error_count=sum(1 for t in tels if t.error),
            agents=[_tel_dict(t) for t in tels],
            tags=list(self.tags),
        )


def _tel_dict(t: AgentTelemetry) -> dict:
    from dataclasses import asdict

    return asdict(t)
