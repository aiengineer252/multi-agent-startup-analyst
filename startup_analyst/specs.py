"""THE CONTRACT — single source of truth for what every agent must produce.

This is the heart of the spec-driven design. One `OutputSpec` per agent, and
it is used in three places at once:

    1. WRITING the prompt    -> spec.render_contract()  (agents.py)
    2. GRADING the output    -> spec.check(report)      (evals.py)
    3. PARSING the result    -> spec.extract_score()    (orchestrator.py)

Because all three read the same object, they can never disagree. Change a
section here and the prompt, the grader, and the parser all update together —
which is exactly the drift you *want* (intentional) instead of the drift you
don't (prompt says one thing, validator checks another).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# The only verdicts the synthesizer is allowed to return. Ordered strongest
# to weakest — used for scoring, reporting, and drift comparison.
VERDICT_CHOICES = ("STRONG GO", "GO WITH CHANGES", "PIVOT", "NO-GO")

# Maps what a model might actually write -> our canonical form.
_VERDICT_ALIASES = {
    "STRONG GO": "STRONG GO",
    "GO WITH CHANGES": "GO WITH CHANGES",
    "NO-GO": "NO-GO",
    "NO GO": "NO-GO",
    "NOGO": "NO-GO",
    "PIVOT": "PIVOT",
}

# Phrases that mean the model dodged the job instead of doing it.
DEFAULT_BANNED = (
    "as an ai language model",
    "i cannot provide",
    "i'm unable to analyze",
    "consult a qualified professional",
    "lorem ipsum",
)


# --------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class Section:
    """One required section of an agent's report.

    `title` is both the instruction the model is given AND the string the
    grader looks for in the output — that's what keeps them in sync.
    """

    title: str
    instruction: str
    required: bool = True


@dataclass(frozen=True)
class RubricCriterion:
    """One dimension the LLM judge scores, 1-5. Weights need not sum to 1;
    they are normalized at scoring time."""

    id: str
    question: str
    weight: float = 1.0


@dataclass(frozen=True)
class Violation:
    """A single contract breach found by spec.check()."""

    rule: str          # machine-readable id, e.g. "missing_section"
    detail: str        # human-readable explanation
    severity: str      # "error" (hard break) | "warn" (soft/style)

    def __str__(self) -> str:
        icon = "✗" if self.severity == "error" else "!"
        return f"{icon} [{self.rule}] {self.detail}"


# --------------------------------------------------------------------- #
# The spec itself
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class OutputSpec:
    agent_key: str
    sections: tuple[Section, ...]

    # Verdict line: specialists end with "**<label>**: ... 7/10"
    verdict_label: Optional[str] = None
    verdict_instruction: str = ""
    score_range: tuple[int, int] = (1, 10)

    # Synthesizer only: must pick one of a fixed set of verdicts.
    verdict_choices: tuple[str, ...] = ()

    # Soft quality gates
    min_words: int = 250
    max_words: int = 2500
    min_numbers: int = 0        # proxy for "showed the math"
    requires_table: bool = False
    banned_phrases: tuple[str, ...] = DEFAULT_BANNED

    rubric: tuple[RubricCriterion, ...] = field(default_factory=tuple)

    # ---------------- 1. Render the contract into the prompt ---------- #

    def render_contract(self) -> str:
        """Turn this spec into the instruction text the agent actually reads.

        The prompt is GENERATED from the spec, never hand-written alongside
        it. That is what makes this spec-driven rather than spec-decorated.
        """
        lines = ["Cover, with headings:"]
        for i, s in enumerate(self.sections, 1):
            lines.append(f"{i}. **{s.title}** — {s.instruction}")

        if self.requires_table:
            lines.append("\nUse a Markdown table where the section calls for one.")

        if self.verdict_choices:
            choices = " / ".join(f"**{v}**" for v in self.verdict_choices)
            lines.append(f"\n{self.verdict_instruction}".rstrip())
            lines.append(f"Your verdict must be exactly one of: {choices}.")
        elif self.verdict_label:
            lo, hi = self.score_range
            lines.append(
                f"\nEnd with a **{self.verdict_label}** line: one sentence, "
                f"then a score written as `N/{hi}` "
                f"({lo}-{hi}) {self.verdict_instruction}".rstrip()
            )
        return "\n".join(lines)

    # ---------------- 2. Parse structured values out of output -------- #

    def extract_score(self, text: str) -> Optional[float]:
        """Pull the numeric score out of a report. Prefers a score that
        appears near the verdict label; falls back to the last one found."""
        _, hi = self.score_range
        pattern = re.compile(
            rf"(\d{{1,2}}(?:\.\d)?)\s*(?:/|\s+out\s+of\s+)\s*{hi}\b", re.I
        )
        matches = list(pattern.finditer(text))

        if self.verdict_label:
            # Anchor on the verdict label if the model wrote it.
            label_re = re.compile(re.escape(self.verdict_label), re.I)
            label_hits = list(label_re.finditer(text))
            if label_hits:
                anchor = label_hits[-1].start()
                after = [m for m in matches if m.start() >= anchor]
                if after:
                    return _clamp(float(after[0].group(1)), self.score_range)

        if matches:
            return _clamp(float(matches[-1].group(1)), self.score_range)

        # Last resort: "Score: 7"
        loose = re.search(r"score[:\s]+(\d{1,2}(?:\.\d)?)\b", text, re.I)
        if loose:
            return _clamp(float(loose.group(1)), self.score_range)
        return None

    def extract_verdict(self, text: str) -> Optional[str]:
        """Pull the categorical verdict (synthesizer only).

        Picks the EARLIEST match in the final-verdict region, breaking ties by
        longest phrase — so 'NO-GO' never gets misread as 'GO', and
        'GO WITH CHANGES' never collapses to 'GO'.
        """
        if not self.verdict_choices:
            return None

        # Prefer the text after the "Final Verdict" heading if present.
        region = text
        anchor = re.search(r"final\s+verdict", text, re.I)
        if anchor:
            region = text[anchor.end():]

        best: Optional[tuple[int, int, str]] = None  # (pos, -len, canonical)
        for alias, canonical in _VERDICT_ALIASES.items():
            for m in re.finditer(rf"\b{re.escape(alias)}\b", region, re.I):
                cand = (m.start(), -len(alias), canonical)
                if best is None or cand < best:
                    best = cand
        return best[2] if best else None

    # ---------------- 3. Grade the output against the contract -------- #

    def check(self, text: str) -> list[Violation]:
        """Deterministic contract check. Free, instant, no API call.

        This is your first line of defence: it catches truncation, refusals,
        skipped sections, and missing scores before you ever pay a judge model.
        """
        v: list[Violation] = []
        low = text.lower()
        words = len(text.split())

        if not text.strip():
            return [Violation("empty_output", "Agent returned nothing.", "error")]

        for s in self.sections:
            if s.required and s.title.lower() not in low:
                v.append(
                    Violation(
                        "missing_section",
                        f"Required section '{s.title}' not found.",
                        "error",
                    )
                )

        if self.verdict_choices:
            if self.extract_verdict(text) is None:
                v.append(
                    Violation(
                        "missing_verdict",
                        "No verdict from "
                        f"{'/'.join(self.verdict_choices)} found.",
                        "error",
                    )
                )
        elif self.verdict_label:
            if self.verdict_label.lower() not in low:
                v.append(
                    Violation(
                        "missing_verdict_line",
                        f"No '{self.verdict_label}' line.",
                        "error",
                    )
                )
            if self.extract_score(text) is None:
                v.append(
                    Violation(
                        "missing_score",
                        f"No parseable score out of {self.score_range[1]}.",
                        "error",
                    )
                )

        if words < self.min_words:
            v.append(
                Violation(
                    "too_short",
                    f"{words} words, expected at least {self.min_words} "
                    "(possible truncation or a lazy answer).",
                    "warn",
                )
            )
        if words > self.max_words:
            v.append(
                Violation(
                    "too_long", f"{words} words, expected at most {self.max_words}.", "warn"
                )
            )

        if self.min_numbers:
            found = len(re.findall(r"\d[\d,.]*", text))
            if found < self.min_numbers:
                v.append(
                    Violation(
                        "insufficient_quantification",
                        f"Only {found} numeric values; this agent must show "
                        f"its math (expected >= {self.min_numbers}).",
                        "warn",
                    )
                )

        if self.requires_table and text.count("|") < 6:
            v.append(
                Violation(
                    "missing_table", "Expected a Markdown table; none found.", "warn"
                )
            )

        for phrase in self.banned_phrases:
            if phrase in low:
                v.append(
                    Violation(
                        "banned_phrase", f"Contains dodge phrase: '{phrase}'.", "error"
                    )
                )
        return v

    def compliance_score(self, text: str) -> float:
        """0.0-1.0. Errors cost 3x what warnings do. Used for dashboards
        and drift, so a single number can be tracked over time."""
        violations = self.check(text)
        if not violations:
            return 1.0
        penalty = sum(3 if x.severity == "error" else 1 for x in violations)
        # Budget scales with how much the contract asks for.
        budget = 3 * (len(self.sections) + 2)
        return max(0.0, 1.0 - penalty / budget)


def _clamp(value: float, bounds: tuple[int, int]) -> float:
    lo, hi = bounds
    return max(float(lo), min(float(hi), value))


# --------------------------------------------------------------------- #
# Shared rubric — what "good" means for every specialist
# --------------------------------------------------------------------- #

BASE_RUBRIC = (
    RubricCriterion(
        "specificity",
        "Is the analysis specific to THIS idea (named companies, concrete "
        "numbers, real channels) rather than generic advice that would apply "
        "to any startup?",
        weight=2.0,
    ),
    RubricCriterion(
        "evidence",
        "Are assumptions stated explicitly and is the reasoning shown, so a "
        "reader could challenge the numbers?",
        weight=2.0,
    ),
    RubricCriterion(
        "decisiveness",
        "Does it commit to a clear view instead of hedging in every direction?",
        weight=1.5,
    ),
    RubricCriterion(
        "calibration",
        "Is confidence proportionate to evidence — neither overclaiming "
        "certainty nor refusing to take a position?",
        weight=1.0,
    ),
    RubricCriterion(
        "actionability",
        "Could a founder do something differently on Monday because of this?",
        weight=1.5,
    ),
)


# --------------------------------------------------------------------- #
# The specs, one per agent
# --------------------------------------------------------------------- #

SPECS: dict[str, OutputSpec] = {
    "market": OutputSpec(
        agent_key="market",
        sections=(
            Section("Target customer", "who exactly buys this, and who uses it."),
            Section(
                "Market size",
                "TAM / SAM / SOM with explicit assumptions and back-of-envelope math.",
            ),
            Section(
                "Market timing & trends",
                "tailwinds, headwinds, and why now (or why not now).",
            ),
            Section(
                "Customer pain intensity",
                "is this a vitamin or a painkiller? What do people do today instead?",
            ),
            Section(
                "Go-to-market channels",
                "the 2-3 most plausible acquisition channels and their difficulty.",
            ),
        ),
        verdict_label="Market Verdict",
        verdict_instruction="for market attractiveness.",
        min_numbers=8,
        rubric=BASE_RUBRIC,
    ),
    "competition": OutputSpec(
        agent_key="competition",
        sections=(
            Section(
                "Direct competitors",
                "likely existing players solving the same problem (name real "
                "companies where you know them; describe archetypes where you don't).",
            ),
            Section(
                "Indirect competitors & substitutes",
                "including the status quo.",
            ),
            Section(
                "Competitive positioning",
                "a plausible wedge/differentiator for this idea, and how durable it is.",
            ),
            Section(
                "Moat analysis",
                "network effects, switching costs, data advantages, brand, economies "
                "of scale: which (if any) apply and on what timeline.",
            ),
            Section(
                "Incumbent response",
                "what happens if a big player copies this in 12 months?",
            ),
        ),
        verdict_label="Competition Verdict",
        verdict_instruction="(10 = weak competition / strong defensibility).",
        min_numbers=3,
        rubric=BASE_RUBRIC,
    ),
    "financials": OutputSpec(
        agent_key="financials",
        sections=(
            Section(
                "Revenue model",
                "the most plausible pricing/monetization, with a specific suggested "
                "price point and why.",
            ),
            Section(
                "Unit economics",
                "estimated CAC, LTV, gross margin, and payback period using labeled "
                "benchmark assumptions; show the math.",
            ),
            Section("Cost structure", "the major cost drivers and how they scale."),
            Section(
                "Path to first revenue",
                "the cheapest credible route to the first $10k MRR (or equivalent).",
            ),
            Section(
                "Funding needs",
                "bootstrappable or venture-scale? Rough capital required to reach "
                "product-market fit.",
            ),
        ),
        verdict_label="Financial Verdict",
        verdict_instruction="for business-model quality.",
        min_numbers=12,
        requires_table=True,
        rubric=BASE_RUBRIC,
    ),
    "risk": OutputSpec(
        agent_key="risk",
        sections=(
            Section(
                "Top 5 risks",
                "a ranked table: risk, category (regulatory / legal / market / "
                "technical / execution / platform), likelihood (L/M/H), impact "
                "(L/M/H), and mitigation.",
            ),
            Section(
                "Regulatory & compliance",
                "licenses, data-privacy (GDPR/DPDP), sector rules, or IP issues "
                "that apply.",
            ),
            Section(
                "Dependency risks",
                "reliance on platforms, APIs, suppliers, or a single acquisition "
                "channel.",
            ),
            Section(
                "Kill criteria",
                "2-3 measurable signals that should make the founder stop or pivot.",
            ),
        ),
        verdict_label="Risk Verdict",
        verdict_instruction="(10 = low overall risk).",
        requires_table=True,
        min_numbers=3,
        rubric=BASE_RUBRIC,
    ),
    "product": OutputSpec(
        agent_key="product",
        sections=(
            Section(
                "Riskiest assumption",
                "the single assumption that, if false, kills the idea, and the "
                "cheapest test for it.",
            ),
            Section(
                "MVP scope",
                "a concrete feature list for v0.1 (what's IN and what's explicitly "
                "OUT), buildable by 1-2 engineers.",
            ),
            Section(
                "Suggested stack & build estimate",
                "a sensible tech stack and an effort estimate in engineer-weeks.",
            ),
            Section(
                "Hard parts",
                "the 2-3 genuinely difficult technical or product problems, and "
                "whether they're solvable by a small team.",
            ),
            Section(
                "Data / AI considerations",
                "if the idea involves ML/AI: data needs, cold-start problem, and "
                "model/API costs.",
            ),
        ),
        verdict_label="Feasibility Verdict",
        verdict_instruction="for buildability.",
        min_numbers=5,
        rubric=BASE_RUBRIC,
    ),
    "memo": OutputSpec(
        agent_key="memo",
        sections=(
            Section(
                "Executive Summary", "3-4 sentences a busy partner reads first."
            ),
            Section(
                "Scorecard",
                "a table with each specialist's score (pull the exact scores from "
                "their verdict lines) plus your overall weighted score /10.",
            ),
            Section("The Bull Case", "the strongest honest argument FOR this idea."),
            Section("The Bear Case", "the strongest honest argument AGAINST it."),
            Section(
                "Where the analysts disagree",
                "contradictions between the reports and your ruling on each.",
            ),
            Section(
                "Recommended next 30 days",
                "3-5 concrete, cheap actions for the founder, ordered by what "
                "retires the most risk per rupee/dollar.",
            ),
            Section(
                "Final Verdict",
                "your call, followed by a 2-3 sentence justification. If GO WITH "
                "CHANGES or PIVOT, state the specific change.",
            ),
        ),
        verdict_choices=VERDICT_CHOICES,
        verdict_instruction=(
            "The Final Verdict section must open with your verdict in bold."
        ),
        requires_table=True,
        min_words=350,
        max_words=3000,
        min_numbers=6,
        rubric=BASE_RUBRIC
        + (
            RubricCriterion(
                "synthesis",
                "Does it genuinely SYNTHESIZE — weighing specialists against each "
                "other and resolving conflicts — rather than just summarizing each "
                "report in turn?",
                weight=2.5,
            ),
            RubricCriterion(
                "consistency",
                "Does the final verdict actually follow from the scorecard and the "
                "bull/bear cases presented?",
                weight=2.0,
            ),
        ),
    ),
}
