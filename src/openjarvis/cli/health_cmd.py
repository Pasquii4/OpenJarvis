"""``jarvis health`` — metrics and system health dashboard."""

import json
import click
from rich.console import Console
from rich.table import Table

from openjarvis.core.config import load_config
from openjarvis.health.metrics import get_health_metrics

@click.command("health", help="Check JARVIS health metrics and status.")
@click.option("--format", "out_format", type=str, default="table", help="Output format: table or json.")
@click.option("--days", type=int, default=7, help="Number of days to analyze.")
def health(out_format: str, days: int) -> None:
    config = load_config()
    
    # Send telemetry event for opening dashboard (can use get_event_bus but this is CLI context)
    console = Console()
    if out_format != "json":
        console.print(f"[dim]Analyzing interactions over the last {days} days...[/dim]")

    telemetry_db = config.telemetry.db_path
    traces_db = config.traces.db_path
    
    metrics = get_health_metrics(telemetry_db_path=telemetry_db, traces_db_path=traces_db, days=days)

    if out_format == "json":
        print(json.dumps(metrics, indent=2))
        return

    table = Table(title=f"JARVIS Health ({days} days)", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value")
    table.add_column("Status")

    # Avg Response Time
    resp = metrics.get("avg_response_time_ms", 0.0)
    if resp > 8000:
        resp_status = "[bold red]Critical[/bold red]"
    elif resp > 3000:
        resp_status = "[bold yellow]Degraded[/bold yellow]"
    else:
        resp_status = "[bold green]OK[/bold green]"
    
    table.add_row("Avg Response Time", f"{resp:.1f} ms", resp_status)

    # Error Rate
    err_rate = metrics.get("error_rate", 0.0)
    if err_rate > 0.25:
        err_status = "[bold red]Critical[/bold red]"
    elif err_rate > 0.10:
        err_status = "[bold yellow]Degraded[/bold yellow]"
    else:
        err_status = "[bold green]OK[/bold green]"

    table.add_row("Error Rate", f"{err_rate*100:.1f} %", err_status)

    # Top Tools Used
    top_tools = metrics.get("top_tools_used", {})
    tools_str = ", ".join([f"{k}({v})" for k,v in top_tools.items()]) if top_tools else "None"
    table.add_row("Top Tools", tools_str, "[dim]-[/dim]")

    # Sessions Per Day (Average approx)
    sessions_avg = sum(metrics.get("sessions_per_day", {}).values()) / days if days else 0
    table.add_row("Avg Sessions/Day", f"{sessions_avg:.1f}", "[dim]-[/dim]")

    # Model Version
    model_version = metrics.get("model_version", "unknown")
    table.add_row("Active Model", model_version, "[dim]-[/dim]")

    console.print(table)
