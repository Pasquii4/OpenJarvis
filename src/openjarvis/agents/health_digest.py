"""Health Digest Agent — synthezises a health briefing in Spanish."""

import json
from datetime import datetime
from typing import Any, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, ToolUsingAgent
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role
from openjarvis.core.config import load_config
from openjarvis.health.metrics import get_health_metrics

@AgentRegistry.register("health_digest")
class HealthDigestAgent(ToolUsingAgent):
    """Generates a weekly text digest summarizing system metrics and goals."""

    agent_id = "health_digest"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._timezone = kwargs.pop("timezone", "Europe/Madrid")
        super().__init__(*args, **kwargs)

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)
        config = load_config()
        
        telemetry_db = config.telemetry.db_path
        traces_db = config.traces.db_path
        
        metrics = get_health_metrics(telemetry_db_path=telemetry_db, traces_db_path=traces_db, days=7)
        now = datetime.now()
        is_sunday = now.weekday() == 6 # 0=Mon, 6=Sun

        # Gather metrics str
        metrics_json = json.dumps(metrics, indent=2)
        
        architect_status = ""
        if is_sunday:
            try:
                # Import check for Architect status locally 
                from openjarvis.cli.architect_cmd import run_architect_checks
                checks_result = run_architect_checks()
                architect_status = "Resultados del check arquitectural:\n"
                for c in checks_result:
                    architect_status += f"- {c['id']}: {c['status']} ({c['detail']})\n"
            except Exception as e:
                architect_status = f"Error integrando architect status: {e}"

        prompt = (
            "Eres un analista de sistemas experto evaluando la salud de JARVIS.\n"
            "Analiza las siguientes métricas de los últimos 7 días y genera un reporte EN ESPAÑOL.\n"
            "El reporte debe contener EXCLUSIVAMENTE texto narrativo, con la siguiente estructura:\n"
            "1. Top 3 problemas detectados (evaluando error rate, latency, etc.).\n"
            "2. Top 3 oportunidades de mejora.\n"
            "3. Comparativa/Resumen del estado.\n"
            "4. Una recomendación de configuración explícita (ej. 'considera usar modelo más rápido').\n\n"
            f"Métricas:\n{metrics_json}\n"
        )
        
        if is_sunday:
            prompt += f"\nHoy es Domingo. Añade al final un apartado 'Estado de Objetivos Arquitecturales':\n{architect_status}"

        messages = [
            Message(role=Role.SYSTEM, content=prompt),
            Message(role=Role.USER, content="Genera el informe de salud ahora.")
        ]
        
        res = self._generate(messages)
        content = self._strip_think_tags(res.get("content", ""))

        self._emit_turn_end(turns=1)
        return AgentResult(
            content=content,
            turns=1,
            metadata={"metrics": metrics, "architect_ran": is_sunday}
        )
