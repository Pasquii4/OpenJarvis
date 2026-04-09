"""``jarvis architect status`` — Verify architectural capability goals."""

import yaml
from pathlib import Path
from typing import List, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

from openjarvis.core.config import load_config
from openjarvis.core.registry import ConnectorRegistry, ToolRegistry, ChannelRegistry

def _check_importable(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def run_architect_checks() -> List[Dict[str, Any]]:
    """Evaluates all architectural capability goals defined in yaml."""
    goals_path = Path("configs/architect_goals.yaml")
    if not goals_path.exists():
        return []

    try:
        with open(goals_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            goals = data.get("goals", [])
    except Exception:
        goals = []

    config = load_config()
    results = []

    for goal in goals:
        status = "❌ Falta"
        detail = "Not implemented"
        
        if goal == "control_home_assistant":
            if ConnectorRegistry.contains("home_assistant") or _check_importable("openjarvis.connectors.home_assistant"):
                status = "✅ OK"
                detail = "Connector present"
        elif goal == "send_whatsapp":
            if any(ChannelRegistry.contains(c) for c in ["whatsapp", "whatsapp_baileys", "sendblue"]):
                status = "✅ OK"
                detail = "WhatsApp channel registered"
        elif goal == "read_pdf_documents":
            if ToolRegistry.contains("file_read"):
                # Simpler check if file_read tool exists, since typically we integrate openjarvis.tools.file_read
                status = "✅ OK"
                detail = "file_read tool present"
        elif goal == "control_spotify":
            if ConnectorRegistry.contains("spotify") or _check_importable("openjarvis.connectors.spotify"):
                status = "✅ OK"
                detail = "Spotify connector present"
        elif goal == "web_browsing_autonomo":
            if ToolRegistry.contains("web_search") and (_check_importable("openjarvis.connectors.browser") or _check_importable("playwright")):
                status = "✅ OK"
                detail = "Web search + browser present"
        elif goal == "github_pr_automation":
            if ConnectorRegistry.contains("github") or ToolRegistry.contains("github"):
                status = "✅ OK"
                detail = "Github integration present"
        elif goal == "calendar_write":
            if ConnectorRegistry.contains("gcalendar"):
                # gcalendar scope validation
                status = "✅ OK"
                detail = "gcalendar connector active"
        elif goal == "email_send":
            if ConnectorRegistry.contains("gmail"):
                status = "✅ OK"
                detail = "gmail integration active"
        elif goal == "notion_integration":
            if ConnectorRegistry.contains("notion") or _check_importable("openjarvis.connectors.notion"):
                status = "✅ OK"
                detail = "Notion connector available"
        elif goal == "voice_cloning":
            if hasattr(config.digest, "custom_voice_model") or getattr(config.digest, "tts_backend", "") == "elevenlabs":
                status = "✅ OK"
                detail = "Custom voice config detected"

        results.append({
            "goal": goal,
            "status": status,
            "detail": detail
        })

    return results

@click.group(name="architect")
def architect_group() -> None:
    """Architectural overview and tracking."""
    pass

@architect_group.command(name="status", help="Check capability goals status.")
@click.option("--format", "out_format", type=str, default="table", help="Format: table or markdown")
def status(out_format: str) -> None:
    results = run_architect_checks()
    
    # Register telemetry event manually to track executions
    try:
        from openjarvis.core.events import get_event_bus, Event, EventType
        bus = get_event_bus()
        bus.publish(Event(
            type=EventType.SYSTEM_INFO,
            source="architect_cmd",
            data={"metric": "architect_status_check", "total_goals": len(results)}
        ))
    except Exception:
        pass

    if out_format == "markdown":
        md = "## Architect Goals Status\n\n| Objetivo | Estado | Detalle |\n|---|---|---|\n"
        for r in results:
            md += f"| {r['goal']} | {r['status']} | {r['detail']} |\n"
        print(md)
        return

    console = Console()
    table = Table(title="Architect Goals Tracker", show_header=True)
    table.add_column("Objetivo", style="cyan")
    table.add_column("Estado")
    table.add_column("Detalle", style="dim")

    for r in results:
        status_str = f"[green]{r['status']}[/green]" if "OK" in r['status'] else f"[red]{r['status']}[/red]"
        table.add_row(r['goal'], status_str, r['detail'])

    console.print(table)
