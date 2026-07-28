"""Core orchestration — provider-agnostic.

Flow:
    1. The five SPECIALISTS run in PARALLEL (one thread each) against the
       same startup idea. Each gets its own persona (system prompt) and task.
    2. Their five Markdown reports are bundled and handed to the SYNTHESIZER,
       which writes the final investment memo with a GO / NO-GO verdict.

All model I/O goes through an LLMProvider (see providers.py), so the same
pipeline runs on Claude, GPT, Groq, or any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from typing import Callable, Optional

from .agents import SPECIALISTS, SYNTHESIZER
from .providers import LLMProvider


@dataclass
class AgentResult:
    key: str
    name: str
    emoji: str
    report: str


class StartupAnalyst:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def analyze(
        self, idea: str, on_status: Optional[Callable[[str], None]] = None
    ) -> list[AgentResult]:
        """Run the full pipeline. Returns results in report order
        (five specialists, then the synthesizer's memo last)."""

        def notify(msg: str) -> None:
            if on_status:
                on_status(msg)

        results: dict[str, AgentResult] = {}

        # ---- Stage 1: specialists in parallel -------------------------- #
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(SPECIALISTS)
        ) as pool:
            futures = {}
            for agent in SPECIALISTS:
                notify(f"{agent['emoji']} {agent['name']} — analyzing...")
                user_prompt = agent["prompt"].format(idea=idea)
                future = pool.submit(
                    self.provider.complete,
                    agent["system"],
                    user_prompt,
                    True,  # specialists may use web search if the provider has it
                )
                futures[future] = agent

            for future in concurrent.futures.as_completed(futures):
                agent = futures[future]
                results[agent["key"]] = AgentResult(
                    key=agent["key"],
                    name=agent["name"],
                    emoji=agent["emoji"],
                    report=future.result(),
                )
                notify(f"{agent['emoji']} {agent['name']} — done ✓")

        ordered = [results[a["key"]] for a in SPECIALISTS]

        # ---- Stage 2: synthesis ---------------------------------------- #
        notify(f"{SYNTHESIZER['emoji']} {SYNTHESIZER['name']} — writing memo...")
        briefing = "\n\n---\n\n".join(f"## {r.name}\n\n{r.report}" for r in ordered)
        memo_prompt = SYNTHESIZER["prompt"].format(idea=idea, reports=briefing)
        memo = self.provider.complete(SYNTHESIZER["system"], memo_prompt, False)
        notify(f"{SYNTHESIZER['emoji']} {SYNTHESIZER['name']} — done ✓")

        ordered.append(
            AgentResult(
                key=SYNTHESIZER["key"],
                name=SYNTHESIZER["name"],
                emoji=SYNTHESIZER["emoji"],
                report=memo,
            )
        )
        return ordered
