"""
ArchitectAgent — Meta-agente de auto-evolución de JARVIS.
Analiza capacidades actuales, detecta gaps y genera prompts de mejora.
"""
from __future__ import annotations
from typing import Any, Optional
from openjarvis.agents._stubs import BaseAgent, AgentContext, AgentResult
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role, _message_to_dict
import json
import re
import yaml
from pathlib import Path
from datetime import datetime

GOALS_FILE = Path("configs/architect_goals.yaml")
PROPOSALS_DIR = Path("src/openjarvis/agents/proposed")

ARCHITECT_SYSTEM_PROMPT = """Eres el Architect de JARVIS — un meta-agente que analiza el sistema y propone mejoras.
Tu tarea es:
1. Analizar las capacidades actuales listadas
2. Compararlas con los objetivos deseados
3. Identificar gaps priorizados por impacto/esfuerzo
4. Para cada gap, generar un prompt autocontenido listo para Antigravity

Responde SIEMPRE en español. Sé directo y técnico. 
Formato de respuesta: Markdown estructurado con secciones:
## Estado Actual
## Gaps Detectados
## Mejoras Priorizadas
## Prompts Antigravity
"""

@AgentRegistry.register("architect")
class ArchitectAgent(BaseAgent):
    agent_id = "architect"

    def _gather_state(self) -> dict:
        state = {}
        
        # 1. Agentes actuales
        state["agents"] = list(AgentRegistry.keys())
        
        # 2. Memoria relevante (últimos 5 fragmentos)
        state["memory_fragments"] = []
        try:
            from openjarvis.memory.jarvis_memory import search_memory
            results = search_memory("errores fallos tareas incompletas", top_k=5)
            state["memory_fragments"] = [r.get("text", "") for r in results]
        except Exception:
            pass

        # 3. Goals
        try:
            if GOALS_FILE.exists():
                with open(GOALS_FILE, "r", encoding="utf-8") as f:
                    y = yaml.safe_load(f)
                    state["goals"] = y.get("goals", [])
            else:
                state["goals"] = [
                    "control_home_assistant",
                    "send_whatsapp",
                    "read_pdf_documents",
                    "control_spotify",
                    "web_browsing_autonomo",
                    "github_pr_automation",
                    "calendar_write",
                    "email_send"
                ]
        except Exception:
            state["goals"] = []

        return state

    def run(self, input: str, context: AgentContext | None = None, **kwargs) -> AgentResult:
        self._emit_turn_start(input)

        state = self._gather_state()
        state_str = json.dumps(state, ensure_ascii=False)
        
        # Build messages
        prompt_interno = f"Estado actual del sistema:\n{state_str}\n\nInput del usuario:\n{input}"
        messages = self._build_messages(prompt_interno, context, system_prompt=ARCHITECT_SYSTEM_PROMPT)

        # Generate response
        result = self._generate(messages, max_tokens=2048, temperature=0.3)
        content = result.get("content", "")
        content = self._strip_think_tags(content)
        
        # Check if scaffold requested
        match = re.search(r'apply\s+(\d+)', input, re.IGNORECASE)
        if match:
            n = int(match.group(1))
            self._scaffold_agent(n, content)

        self._emit_turn_end()
        return AgentResult(
            content=content,
            metadata={"state": state, "timestamp": datetime.now().isoformat()}
        )

    def _scaffold_agent(self, n: int, analysis: str) -> str:
        PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
        
        import datetime
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        
        # Try to parse agent name and description from the analysis text
        # Usually looking for "## Mejoras Priorizadas" or "N. Nombre" etc.
        agent_name = f"proposed_agent_{n}"
        desc = "Agente generado automáticamente."
        
        # Rough extraction attempt
        pattern = rf"(?:###?|[-*•]\s+)?\s*{n}[\.\)]?\s*\**([a-zA-Z0-9_]+)\**"
        m = re.search(pattern, analysis)
        if m:
            agent_name = m.group(1).lower()

        cls_name = "".join(x.capitalize() for x in agent_name.split("_"))
        
        file_path = PROPOSALS_DIR / f"agente_{date_str}_{n}.py"
        code = f'''"""{agent_name} — generado por ArchitectAgent el {date_str}"""
from openjarvis.agents._stubs import BaseAgent, AgentContext, AgentResult
from openjarvis.core.registry import AgentRegistry

@AgentRegistry.register("{agent_name}")
class {cls_name}(BaseAgent):
    agent_id = "{agent_name}"
    
    def run(self, input: str, context: AgentContext | None = None, **kwargs) -> AgentResult:
        # TODO: implementar lógica del agente
        # Descripción: {desc}
        raise NotImplementedError
'''
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        return str(file_path)

__all__ = ["ArchitectAgent", "ARCHITECT_SYSTEM_PROMPT"]
