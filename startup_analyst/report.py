"""Assemble agent results into a single Markdown report and save it."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .orchestrator import AgentResult


def build_markdown(idea: str, results: list[AgentResult], model: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    memo = results[-1]          # synthesizer is always last
    specialists = results[:-1]

    parts = [
        "# Startup Analysis Report",
        "",
        f"> **Generated:** {timestamp}  ",
        f"> **Model:** `{model}`  ",
        f"> **Agents:** {len(specialists)} specialists + 1 synthesizer",
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
        "# Appendix: Specialist Reports",
        "",
    ]
    for r in specialists:
        parts += [f"## {r.emoji} {r.name}", "", r.report, "", "---", ""]

    return "\n".join(parts)


def save_report(markdown: str, idea: str, output_dir: str = "reports") -> Path:
    """Write the report to reports/<slug>_<timestamp>.md and return the path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^a-z0-9]+", "-", idea.lower()).strip("-")[:40] or "analysis"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"{slug}_{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    return path
