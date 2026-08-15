"""Multi-Agent Startup Analyst — CLI entry point.

Usage:
    python main.py "Your startup idea here"
    python main.py --file examples/sample_idea.txt
    python main.py "idea" --web                        # Claude + live web search
    python main.py "idea" --provider groq              # Groq (fast + free tier)
    python main.py "idea" --provider openai -m gpt-5.1 # OpenAI GPT
    python main.py "idea" --provider custom \
        --base-url http://localhost:11434/v1 -m llama3.1:8b   # local Ollama

Set the matching API key in .env (see .env.example):
    anthropic -> ANTHROPIC_API_KEY   openai -> OPENAI_API_KEY
    groq      -> GROQ_API_KEY        custom -> CUSTOM_API_KEY (+ CUSTOM_BASE_URL)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from startup_analyst.monitoring import resolve_log_path
from startup_analyst.orchestrator import StartupAnalyst
from startup_analyst.providers import (
    PROVIDER_PRESETS,
    ProviderError,
    create_provider,
)
from startup_analyst.report import build_markdown, save_report

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="startup-analyst",
        description="A team of AI agents that evaluates a startup idea and "
        "produces an investment memo with a GO / NO-GO verdict.",
    )
    parser.add_argument("idea", nargs="?", help="The startup idea, in quotes.")
    parser.add_argument(
        "--file", "-f", help="Read the idea from a text file instead."
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=list(PROVIDER_PRESETS),
        default="anthropic",
        help="Which LLM API to use (default: anthropic). 'custom' works with "
        "any OpenAI-compatible endpoint (Ollama, OpenRouter, Together, ...).",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model name. Defaults per provider: "
        + ", ".join(
            f"{name}={preset['default_model'] or '(required)'}"
            for name, preset in PROVIDER_PRESETS.items()
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API endpoint for --provider custom "
        "(e.g. http://localhost:11434/v1 for Ollama).",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Name of the environment variable holding the API key "
        "(overrides the provider's default, e.g. OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Let agents use live web search for fresher market/competitor data. "
        "Anthropic provider only (slower, costs extra per search).",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="reports",
        help="Directory for the saved Markdown report (default: reports/).",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Skip printing the full memo to the terminal (just save the file).",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Tag this run in the log (repeatable), e.g. --tag experiment-b. "
        "Useful for A/B comparing prompts or models later.",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Run-log path (default: runs/runs.jsonl, or $ANALYST_RUN_LOG).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Don't record this run in the run log.",
    )
    return parser.parse_args()


def get_idea(args: argparse.Namespace) -> str:
    if args.file:
        try:
            with open(args.file, encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError as exc:
            console.print(f"[red]Could not read {args.file}: {exc}[/red]")
            sys.exit(1)
    if args.idea:
        return args.idea.strip()
    console.print(
        "[red]No idea provided.[/red] Pass it as an argument or with --file.\n"
        'Example:  python main.py "An app that matches home cooks with nearby buyers"'
    )
    sys.exit(1)


def main() -> None:
    load_dotenv()
    args = parse_args()
    idea = get_idea(args)

    use_web = args.web
    if use_web and args.provider != "anthropic":
        console.print(
            "[yellow]Note:[/yellow] --web (live web search) is only available "
            "on the anthropic provider — continuing without it."
        )
        use_web = False

    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[yellow]Warning:[/yellow] ANTHROPIC_API_KEY is not set. "
            "Create a .env file (copy .env.example) or export the variable.\n"
            "Continuing anyway in case another credential source is configured..."
        )

    try:
        provider = create_provider(
            args.provider,
            model=args.model,
            use_web_search=use_web,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
    except ProviderError as exc:
        console.print(f"[red]Provider setup failed:[/red] {exc}")
        sys.exit(1)

    console.print(
        Panel.fit(
            f"[bold]Idea:[/bold] {idea[:300]}{'...' if len(idea) > 300 else ''}\n"
            f"[bold]Provider:[/bold] {provider.label}   "
            f"[bold]Web search:[/bold] {'on' if use_web else 'off'}",
            title="🚀 Multi-Agent Startup Analyst",
            border_style="cyan",
        )
    )

    analyst = StartupAnalyst(
        provider,
        log_path=str(resolve_log_path(args.log)),
        tags=args.tag or [],
    )

    start = time.time()
    try:
        run = analyst.analyze(
            idea,
            on_status=lambda m: console.print(f"  {m}"),
            persist=not args.no_log,
        )
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        sys.exit(130)
    except Exception as exc:  # auth errors, rate limits, network, etc.
        console.print(f"\n[red]Analysis failed:[/red] {exc}")
        console.print(
            "[dim]Check your API key, network connection, and account credits. "
            "Rate-limited? Wait a minute and retry, or try a smaller/faster "
            "model (e.g. --model claude-haiku-4-5, or --provider groq).[/dim]"
        )
        sys.exit(1)
    elapsed = time.time() - start
    record = run.record

    markdown = build_markdown(idea, run.results, provider.label, record)
    path = save_report(markdown, idea, args.out)

    cost = (
        f"${record.total_cost_usd:.4f}"
        if record.total_cost_usd is not None
        else "cost n/a"
    )
    console.print(
        f"\n[green]✓ Analysis complete in {elapsed:.0f}s[/green] — "
        f"{cost} · {record.total_output_tokens:,} output tokens · "
        f"contract compliance {record.mean_compliance:.0%}"
    )
    console.print(f"  Report saved to [bold]{path}[/bold]")

    # Surface quality problems immediately rather than burying them in the log.
    if record.error_count:
        console.print(
            f"  [red]⚠ {record.error_count} agent(s) failed[/red] — "
            "the memo was written from partial input."
        )
    off_contract = [
        a for a in record.agents if a["violations"] and not a["error"]
    ]
    if off_contract:
        console.print(
            f"  [yellow]⚠ {len(off_contract)} agent(s) broke their output "
            "contract[/yellow] (see Appendix B in the report)"
        )

    if not args.no_print:
        memo = run.memo
        console.print(
            Panel(
                Markdown(memo.report),
                title=f"{memo.emoji} Final Investment Memo",
                border_style="green",
            )
        )
        console.print(
            f"\n[dim]Full report (memo + 5 specialist analyses + telemetry): "
            f"{path}[/dim]"
        )
        if not args.no_log:
            console.print(
                "[dim]Run logged. See trends with: python monitor.py dashboard[/dim]"
            )


if __name__ == "__main__":
    main()
