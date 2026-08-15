"""Assemble agent results into a single Markdown report and save it.

The report has three layers, in decreasing order of who reads them:
    1. The memo          — what a founder or partner actually reads
    2. Specialist reports — the working papers behind the memo
    3. Run telemetry      — what it cost, how long it took, and whether every
                            agent honoured its contract

Layer 3 matters: a memo built from a report that silently skipped two required
sections should not look identical to a clean one. Putting the compliance
numbers in the artifact itself means quality problems travel with the output
instead of hiding in a log file.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .monitoring import RunRecord
from .orchestrator import AgentResult


def build_markdown(
    idea: str,
    results: list[AgentResult],
    model: str,
    record: Optional[RunRecord] = None,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    memo = results[-1]          # synthesizer is always last
    specialists = results[:-1]

    parts = [
        "# Startup Analysis Report",
        "",
        f"> **Generated:** {timestamp}  ",
        f"> **Model:** `{model}`  ",
        f"> **Agents:** {len(specialists)} specialists + 1 synthesizer",
    ]
    if record:
        cost = (
            f"${record.total_cost_usd:.4f}"
            if record.total_cost_usd is not None
            else "n/a"
        )
        parts.append(
            f"> **Run:** `{record.run_id}` · {record.total_latency_s:.0f}s · "
            f"{cost} · contract compliance {record.mean_compliance:.0%}"
        )
    parts += [
        "",
        "## The Idea",
        "",
        f"> {idea.strip()}",
        "",
        "---",
        "",
        f"# {memo.emoji} Investment Memo — {memo.name}",
        "",
        memo.report,
        "",
        "---",
        "",
        "# Appendix A: Specialist Reports",
        "",
    ]
    for r in specialists:
        parts += [f"## {r.emoji} {r.name}", "", r.report, "", "---", ""]

    if record:
        parts += _telemetry_section(record)

    return "\n".join(parts)


def _telemetry_section(record: RunRecord) -> list[str]:
    """A compact audit trail: what each agent cost and whether it complied."""
    lines = [
        "# Appendix B: Run Telemetry",
        "",
        "| Agent | Time | Tokens (in/out) | Cost | Score | Contract |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in record.agents:
        cost = f"${a['cost_usd']:.4f}" if a.get("cost_usd") is not None else "n/a"
        score = f"{a['score']:.0f}" if a.get("score") is not None else "—"
        lines.append(
            f"| {a['name']} | {a['latency_s']:.1f}s | "
            f"{a['input_tokens']:,}/{a['output_tokens']:,} | {cost} | {score} | "
            f"{a['compliance']:.0%} |"
        )

    total_cost = (
        f"${record.total_cost_usd:.4f}"
        if record.total_cost_usd is not None
        else "n/a"
    )
    lines += [
        f"| **Total** | **{record.total_latency_s:.1f}s** | "
        f"**{record.total_input_tokens:,}/{record.total_output_tokens:,}** | "
        f"**{total_cost}** | | **{record.mean_compliance:.0%}** |",
        "",
    ]

    violations = [
        (a["name"], v) for a in record.agents for v in a.get("violations", [])
    ]
    if violations:
        lines += ["## Contract violations", ""]
        lines += [f"- **{name}** — {v}" for name, v in violations]
        lines.append("")
    else:
        lines += ["_All agents satisfied their output contracts._", ""]

    errors = [(a["name"], a["error"]) for a in record.agents if a.get("error")]
    if errors:
        lines += ["## Agent failures", ""]
        lines += [f"- **{name}** — `{err}`" for name, err in errors]
        lines.append("")

    return lines


def save_report(markdown: str, idea: str, output_dir: str = "reports") -> Path:
    """Write the report to reports/<slug>_<timestamp>.md and return the path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^a-z0-9]+", "-", idea.lower()).strip("-")[:40] or "analysis"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"{slug}_{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    return path
