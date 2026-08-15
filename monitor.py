"""Inspect what the system has been doing: dashboard, baselines, drift.

    python monitor.py dashboard              # health summary of recent runs
    python monitor.py runs --limit 20        # the raw run list
    python monitor.py baseline save          # freeze current metrics as reference
    python monitor.py baseline list
    python monitor.py drift                  # compare recent window vs baseline

Typical loop:
    1. Get the system to a state you're happy with.
    2. `python monitor.py baseline save --name v1`
    3. Keep using it normally; every run is logged automatically.
    4. `python monitor.py drift --baseline v1` whenever you want to check,
       or in CI after any prompt/spec/model change.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from startup_analyst.drift import (
    DRIFT,
    OK,
    WARN,
    compare,
    list_baselines,
    load_baseline,
    save_baseline,
)
from startup_analyst.monitoring import (
    filter_runs,
    load_runs,
    resolve_log_path,
    summarize,
)

console = Console()

STATUS_COLOR = {OK: "green", WARN: "yellow", DRIFT: "red", "UNKNOWN": "dim"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="monitor", description="Monitoring and drift detection."
    )
    p.add_argument("--log", default=None, help="Run-log path (default runs/runs.jsonl).")
    p.add_argument("--provider", default=None, help="Filter to one provider.")
    p.add_argument("--model", default=None, help="Filter to one model.")
    p.add_argument("--tag", default=None, help="Filter to one tag (e.g. 'eval').")
    p.add_argument("--include-evals", action="store_true",
                   help="Include eval runs (excluded by default — their ideas are "
                        "deliberately bad and would skew the metrics).")

    sub = p.add_subparsers(dest="command")
    sub.add_parser("dashboard", help="Health summary (default).")

    runs_p = sub.add_parser("runs", help="List recent runs.")
    runs_p.add_argument("--limit", type=int, default=15)

    base_p = sub.add_parser("baseline", help="Manage baselines.")
    base_sub = base_p.add_subparsers(dest="baseline_command")
    save_p = base_sub.add_parser("save", help="Freeze current metrics.")
    save_p.add_argument("--name", default="default")
    save_p.add_argument("--window", type=int, default=None,
                        help="Use only the last N runs.")
    base_sub.add_parser("list", help="List saved baselines.")

    drift_p = sub.add_parser("drift", help="Compare recent runs to a baseline.")
    drift_p.add_argument("--baseline", default="default")
    drift_p.add_argument("--window", type=int, default=10,
                         help="How many recent runs to compare (default 10).")
    drift_p.add_argument("--fail-on", choices=["warn", "drift", "never"],
                         default="drift",
                         help="Exit non-zero at this severity (for CI).")
    return p.parse_args()


def get_runs(args) -> list[dict]:
    runs = load_runs(resolve_log_path(args.log))
    if not args.include_evals and not args.tag:
        runs = [r for r in runs if "eval" not in r.get("tags", [])]
    return filter_runs(runs, provider=args.provider, model=args.model, tag=args.tag)


def fmt(value, spec="", dash="—"):
    return f"{value:{spec}}" if value is not None else dash


# --------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------- #

def cmd_dashboard(args) -> None:
    runs = get_runs(args)
    if not runs:
        console.print(
            "[yellow]No runs logged yet.[/yellow] Run an analysis first:\n"
            '  python main.py "some idea" --provider mock'
        )
        return

    s = summarize(runs)
    console.print(
        Panel.fit(
            f"[bold]{s['count']}[/bold] runs   "
            f"{s['first'][:10]} → {s['last'][:10]}\n"
            f"Providers: {', '.join(s['providers'])}\n"
            f"Models: {', '.join(s['models'])}",
            title="📈 Run Health",
            border_style="cyan",
        )
    )

    t = Table(header_style="bold", title="Overall")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_row("Mean analyst score", fmt(s["mean_score"], ".2f") + " /10")
    t.add_row("Mean spec compliance", fmt(s["mean_compliance"], ".3f"))
    t.add_row("Judge quality", fmt(s["judge_overall"], ".2f") + " /5"
              if s["judge_overall"] else "— (run eval.py --judge)")
    t.add_row("Latency p50", fmt(s["latency_p50_s"], ".1f") + "s")
    t.add_row("Latency p95", fmt(s["latency_p95_s"], ".1f") + "s")
    t.add_row("Cost / run", "$" + fmt(s["cost_mean_usd"], ".4f")
              if s["cost_mean_usd"] is not None else "n/a")
    t.add_row("Cost total", "$" + fmt(s["cost_total_usd"], ".3f")
              if s["cost_total_usd"] is not None else "n/a")
    t.add_row("Runs with a failed agent", str(s["error_runs"]))
    console.print(t)

    vt = Table(header_style="bold", title="Verdict mix")
    vt.add_column("Verdict")
    vt.add_column("Runs", justify="right")
    vt.add_column("Share", justify="right")
    total = sum(s["verdicts"].values())
    for verdict, n in sorted(s["verdicts"].items(), key=lambda x: -x[1]):
        vt.add_row(verdict, str(n), f"{n/total:.0%}")
    console.print(vt)

    at = Table(header_style="bold", title="Per-agent")
    at.add_column("Agent")
    at.add_column("Latency", justify="right")
    at.add_column("Cost", justify="right")
    at.add_column("Compliance", justify="right")
    at.add_column("Score", justify="right")
    at.add_column("Violations", justify="right")
    at.add_column("Errors", justify="right")
    for _, a in s["agents"].items():
        comp = a["mean_compliance"] or 0
        color = "green" if comp >= 0.95 else "yellow" if comp >= 0.8 else "red"
        at.add_row(
            a["name"],
            fmt(a["mean_latency_s"], ".1f") + "s",
            "$" + fmt(a["mean_cost_usd"], ".4f")
            if a["mean_cost_usd"] is not None else "n/a",
            f"[{color}]{comp:.2f}[/{color}]",
            fmt(a["mean_score"], ".1f"),
            str(a["total_violations"]),
            str(a["errors"]),
        )
    console.print(at)


def cmd_runs(args) -> None:
    runs = get_runs(args)[-args.limit:]
    if not runs:
        console.print("[yellow]No runs logged yet.[/yellow]")
        return
    t = Table(header_style="bold", title=f"Last {len(runs)} runs")
    t.add_column("When")
    t.add_column("Model")
    t.add_column("Idea", max_width=34)
    t.add_column("Verdict")
    t.add_column("Score", justify="right")
    t.add_column("Spec", justify="right")
    t.add_column("Cost", justify="right")
    t.add_column("Time", justify="right")
    for r in runs:
        t.add_row(
            r["timestamp"][5:16].replace("T", " "),
            r["model"][:18],
            r["idea_preview"][:34],
            r.get("verdict") or "—",
            fmt(r.get("mean_score"), ".1f"),
            f"{r.get('mean_compliance', 0):.2f}",
            "$" + fmt(r.get("total_cost_usd"), ".4f")
            if r.get("total_cost_usd") is not None else "n/a",
            f"{r['total_latency_s']:.0f}s",
        )
    console.print(t)


def cmd_baseline(args) -> None:
    if args.baseline_command == "list":
        names = list_baselines()
        if not names:
            console.print("[yellow]No baselines saved yet.[/yellow] "
                          "Create one with: python monitor.py baseline save")
            return
        for n in names:
            b = load_baseline(n)
            console.print(
                f"  [bold]{n}[/bold] — {b['run_count']} runs, "
                f"created {b['created_at'][:10]}"
            )
        return

    runs = get_runs(args)
    if not runs:
        console.print("[red]No runs to baseline.[/red] Run some analyses first.")
        sys.exit(1)
    if args.window:
        runs = runs[-args.window:]
    path = save_baseline(runs, name=args.name)
    console.print(
        f"[green]✓ Baseline '{args.name}' saved[/green] from {len(runs)} runs → {path}\n"
        "[dim]Commit this file to git so prompt changes show up as baseline "
        "changes in review.[/dim]"
    )


def cmd_drift(args) -> None:
    baseline = load_baseline(args.baseline)
    if not baseline:
        console.print(
            f"[red]No baseline named '{args.baseline}'.[/red]\n"
            "Create one with: python monitor.py baseline save --name "
            f"{args.baseline}"
        )
        sys.exit(1)

    runs = get_runs(args)[-args.window:]
    if not runs:
        console.print("[yellow]No recent runs to compare.[/yellow]")
        sys.exit(0)

    report = compare(baseline, runs)
    color = STATUS_COLOR[report.status]
    console.print(
        Panel.fit(
            f"Baseline [bold]{report.baseline_name}[/bold] "
            f"({report.baseline_runs} runs)  vs  "
            f"last [bold]{report.current_runs}[/bold] runs\n"
            f"Status: [{color} bold]{report.status}[/{color} bold]   "
            f"Confidence: {report.confidence}",
            title="🔍 Drift Report",
            border_style=color,
        )
    )
    if report.confidence == "LOW":
        console.print(
            "[dim]LOW confidence: too few runs for these differences to be "
            "meaningful. Treat any flag as 'worth a look', not proof.[/dim]\n"
        )

    t = Table(header_style="bold")
    t.add_column("Signal")
    t.add_column("Kind")
    t.add_column("Status", justify="center")
    t.add_column("Change")
    for s in report.signals:
        c = STATUS_COLOR[s.status]
        t.add_row(s.metric, s.kind, f"[{c}]{s.status}[/{c}]", s.note)
    console.print(t)

    if report.problems:
        console.print("\n[bold]What to check:[/bold]")
        hints = {
            "quality": "Did a prompt, spec, or model version change?",
            "behaviour": "The system is reaching different conclusions — "
                         "compare a golden case run before/after.",
            "cost": "Longer outputs or a pricier model. Check per-agent costs "
                    "in `monitor.py dashboard`.",
            "latency": "Provider slowness, bigger prompts, or web search on.",
            "reliability": "Agents are erroring — check rate limits and keys.",
        }
        seen = set()
        for s in report.problems:
            if s.kind not in seen:
                console.print(f"  • [bold]{s.kind}[/bold]: {hints[s.kind]}")
                seen.add(s.kind)

    if args.fail_on == "warn" and report.status in (WARN, DRIFT):
        sys.exit(1)
    if args.fail_on == "drift" and report.status == DRIFT:
        sys.exit(1)


def main() -> None:
    load_dotenv()
    args = parse_args()
    command = args.command or "dashboard"
    if command == "dashboard":
        cmd_dashboard(args)
    elif command == "runs":
        cmd_runs(args)
    elif command == "baseline":
        if not getattr(args, "baseline_command", None):
            console.print("Usage: python monitor.py baseline [save|list]")
            sys.exit(1)
        cmd_baseline(args)
    elif command == "drift":
        cmd_drift(args)


if __name__ == "__main__":
    main()
