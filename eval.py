"""Run the evaluation suite.

    python eval.py                          # all golden cases, no judge (free-ish)
    python eval.py --judge                  # + LLM-as-judge quality scoring
    python eval.py --provider mock          # offline smoke test, zero cost
    python eval.py --case obvious_dud       # one case only
    python eval.py --json results.json      # machine-readable output for CI

Exit code is 0 when every case passes and 1 when any fails, so this drops
straight into CI:  `python eval.py --provider groq || exit 1`
"""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from startup_analyst.evals import as_dict, run_suite
from startup_analyst.golden_cases import CASES_BY_ID, GOLDEN_CASES
from startup_analyst.providers import PROVIDER_PRESETS, ProviderError, create_provider

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval",
        description="Grade the analyst pipeline against golden test cases.",
    )
    p.add_argument("--provider", "-p", choices=list(PROVIDER_PRESETS),
                   default="anthropic", help="Provider under test.")
    p.add_argument("--model", "-m", default=None, help="Model under test.")
    p.add_argument("--base-url", default=None, help="For --provider custom.")
    p.add_argument("--api-key-env", default=None, help="Override key env var.")
    p.add_argument("--case", "-c", action="append", default=None,
                   help="Run only this case id (repeatable). "
                        f"Available: {', '.join(CASES_BY_ID)}")
    p.add_argument("--judge", action="store_true",
                   help="Also run LLM-as-judge quality scoring (extra API calls).")
    p.add_argument("--judge-provider", default=None,
                   help="Provider for the judge (default: same as --provider).")
    p.add_argument("--judge-model", default=None,
                   help="Model for the judge. Use a strong model here.")
    p.add_argument("--json", dest="json_out", default=None,
                   help="Write full results to this JSON file.")
    p.add_argument("--log", default=None, help="Run-log path override.")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    cases = GOLDEN_CASES
    if args.case:
        unknown = [c for c in args.case if c not in CASES_BY_ID]
        if unknown:
            console.print(f"[red]Unknown case id(s):[/red] {', '.join(unknown)}")
            console.print(f"Available: {', '.join(CASES_BY_ID)}")
            sys.exit(1)
        cases = tuple(CASES_BY_ID[c] for c in args.case)

    try:
        provider = create_provider(
            args.provider, model=args.model,
            base_url=args.base_url, api_key_env=args.api_key_env,
        )
        judge = None
        if args.judge:
            judge = create_provider(
                args.judge_provider or args.provider,
                model=args.judge_model,
                base_url=args.base_url,
            )
    except ProviderError as exc:
        console.print(f"[red]Provider setup failed:[/red] {exc}")
        sys.exit(1)

    console.print(
        Panel.fit(
            f"[bold]Under test:[/bold] {provider.label}\n"
            f"[bold]Cases:[/bold] {len(cases)}   "
            f"[bold]Judge:[/bold] {judge.label if judge else 'off'}",
            title="🧪 Evaluation Suite",
            border_style="cyan",
        )
    )

    suite = run_suite(
        provider, cases=cases, judge=judge,
        on_status=lambda m: console.print(f"  {m}"), log_path=args.log,
    )
    summary = suite.summary()

    # ---- results table -------------------------------------------------
    table = Table(title="\nCase results", header_style="bold")
    table.add_column("Case")
    table.add_column("Result", justify="center")
    table.add_column("Verdict")
    table.add_column("Score", justify="right")
    table.add_column("Spec", justify="right")
    if args.judge:
        table.add_column("Judge", justify="right")
    table.add_column("Latency", justify="right")

    for c in suite.cases:
        row = [
            c.case_id,
            "[green]PASS[/green]" if c.passed else "[red]FAIL[/red]",
            c.verdict or "—",
            f"{c.mean_score:.1f}" if c.mean_score is not None else "—",
            f"{c.mean_compliance:.2f}",
        ]
        if args.judge:
            row.append(f"{c.judge_overall:.2f}" if c.judge_overall else "—")
        row.append(f"{c.latency_s:.0f}s")
        table.add_row(*row)
    console.print(table)

    # ---- failures ------------------------------------------------------
    if summary["failures"]:
        console.print("\n[red bold]Failed assertions[/red bold]")
        for case_id, fails in summary["failures"].items():
            console.print(f"\n  [bold]{case_id}[/bold]")
            for f in fails:
                console.print(f"    [red]{f}[/red]")

    cost = summary.get("total_cost_usd")
    console.print(
        f"\n[bold]{summary['passed']}/{summary['cases']} cases passed[/bold]  "
        f"(pass rate {summary['pass_rate']:.0%}, "
        f"mean spec compliance {summary['mean_compliance']:.2f}"
        + (f", judge {summary['judge_overall']:.2f}/5" if summary["judge_overall"] else "")
        + (f", cost ${cost:.3f}" if cost else "")
        + ")"
    )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(as_dict(suite), fh, indent=2)
        console.print(f"[dim]Full results written to {args.json_out}[/dim]")

    if suite.passed:
        console.print("[green]✓ Suite passed[/green]")
    else:
        console.print("[red]✗ Suite failed[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
