"""Morning Digest Agent — synthesizes a daily briefing from multiple sources.

Thin orchestrator that delegates to digest_collect (data fetching),
the LLM (narrative synthesis), and text_to_speech (audio generation).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, ToolUsingAgent
from openjarvis.agents.digest_store import DigestArtifact, DigestStore
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role, ToolCall


def _load_persona(persona_name: str) -> str:
    """Load a persona prompt file by name."""
    search_paths = [
        Path("configs/openjarvis/prompts/personas") / f"{persona_name}.md",
        Path.home() / ".openjarvis" / "prompts" / "personas" / f"{persona_name}.md",
    ]
    for p in search_paths:
        if p.exists():
            return p.read_text(encoding="utf-8")
    return ""


@AgentRegistry.register("morning_digest")
class MorningDigestAgent(ToolUsingAgent):
    """Pre-compute a daily digest from configured data sources."""

    agent_id = "morning_digest"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Extract digest-specific kwargs before passing to parent
        self._persona = kwargs.pop("persona", "jarvis")
        self._sections = kwargs.pop(
            "sections", ["messages", "calendar", "world"]
        )
        self._section_sources = kwargs.pop("section_sources", {})
        self._timezone = kwargs.pop("timezone", "Europe/Madrid")
        self._voice_id = kwargs.pop("voice_id", "es-ES-AlvaroNeural")
        self._voice_speed = kwargs.pop("voice_speed", 1.0)
        self._tts_backend = kwargs.pop("tts_backend", "edge-tts")
        self._digest_store_path = kwargs.pop("digest_store_path", "")
        self._honorific = kwargs.pop("honorific", "Pau")
        super().__init__(*args, **kwargs)

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt from persona + briefing structure."""
        persona_text = _load_persona(self._persona)
        now = datetime.now()
        honorific = getattr(self, "_honorific", "Pau")

        return (
            f"{persona_text}\n\n"
            f"Hoy es {now.strftime('%A, %d de %B de %Y')}. "
            f"Son las {now.strftime('%H:%M')} en {self._timezone}.\n"
            f"El usuario se llama: {honorific}\n\n"
            "Recibes datos estructurados de los servicios conectados del usuario. "
            "Los datos YA han sido recopilados — aparecen en el mensaje del "
            "usuario. NO recoges nada tú mismo.\n\n"
            "Produce un briefing matutino hablado de 2-4 minutos en orden "
            "DECRECIENTE de importancia:\n\n"
            "1. SALUDO + PRIORIDADES — Abre con el nombre del usuario e "
            "indica inmediatamente qué necesita atención: tareas vencidas, "
            "plazos de hoy, eventos que requieren preparación. Conecta "
            "elementos relacionados.\n\n"
            "2. AGENDA — Eventos de hoy con contexto temporal: 'Tienes "
            "3 horas antes de tu próxima reunión.' Omite eventos pasados.\n\n"
            "3. MENSAJES — Triaje entre TODOS los canales (email, mensajes):\n"
            "  - Primero: mensajes de personas reales que necesitan RESPUESTA\n"
            "  - Segundo: mensajes con plazos o elementos de acción\n"
            "  - Último: mención breve de hilos casuales\n"
            "  - OMITE emails automatizados, newsletters y marketing\n\n"
            "4. TIEMPO — Previsión meteorológica para hoy.\n\n"
            "5. RESUMEN — Una o dos frases con lo más importante del día.\n\n"
            "REGLAS ABSOLUTAS (las violaciones son inaceptables):\n"
            "- SOLO hechos de los datos proporcionados. Cero invención.\n"
            "- NUNCA menciones fuentes desconectadas o no disponibles.\n"
            "- NUNCA describas acciones que estás realizando.\n"
            "- Reconoce cada fuente que devolvió datos, aunque sea brevemente.\n"
            "- Sin markdown, emojis en el texto hablado, viñetas ni encabezados.\n"
            "- LÍMITE ESTRICTO: 200 palabras. Sé conciso."
        )

    def _resolve_sources(self) -> List[str]:
        """Get the list of connector IDs to query."""
        default_source_map = {
            "messages": [
                "gmail",
                "slack",
                "google_tasks",
                "imessage",
                "github_notifications",
            ],
            "calendar": ["gcalendar"],
            "health": ["oura", "apple_health"],
            "world": ["weather", "hackernews", "news_rss"],
            "music": ["spotify", "apple_music"],
        }
        sources = set()
        for section in self._sections:
            section_sources = self._section_sources.get(
                section, default_source_map.get(section, [])
            )
            sources.update(section_sources)
        return list(sources)

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)

        # Step 1: Collect data from connectors
        sources = self._resolve_sources()
        collect_call = ToolCall(
            id="digest-collect-1",
            name="digest_collect",
            arguments=json.dumps({"sources": sources, "hours_back": 24}),
        )
        collect_result = self._executor.execute(collect_call)
        collected_data = collect_result.content

        # Step 2: Synthesize narrative via LLM
        system_prompt = self._build_system_prompt()
        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(
                role=Role.USER,
                content=(
                    f"Here is the collected data from my sources:\n\n"
                    f"{collected_data}\n\n"
                    f"Synthesize my morning briefing. Remember:\n"
                    f"- Priority-first, connect related items\n"
                    f"- For health: say 'solid', 'improving', 'dipped' "
                    f"— NEVER say any number (no 82, no 56, no 6000)\n"
                    f"- Do NOT invent reasons for health changes\n"
                    f"- Do NOT mention disconnected sources\n"
                    f"- Do NOT repeat the greeting in your closing\n"
                    f"- Use the honorific ONLY 2-3 times total\n"
                    f"- Skip notifications from the user themselves\n"
                    f"- STRICT LIMIT: 200-250 words maximum"
                ),
            ),
        ]

        result = self._generate(messages)
        narrative = self._strip_think_tags(result.get("content", ""))

        # Step 2b: Self-evaluate and optionally regenerate
        quality_score = 0.0
        evaluator_feedback = ""
        try:
            from openjarvis.agents.digest_evaluator import DigestEvaluator

            evaluator = DigestEvaluator(self._engine, self._model)
            quality_score, evaluator_feedback = evaluator.evaluate(
                collected_data, narrative
            )

            if quality_score < 7.0 and evaluator_feedback:
                # Regenerate with feedback
                messages.append(
                    Message(
                        role=Role.USER,
                        content=(
                            f"Your briefing scored {quality_score:.1f}/10. "
                            f"Feedback: {evaluator_feedback}\n"
                            f"Please revise the briefing addressing this feedback."
                        ),
                    )
                )
                result = self._generate(messages)
                narrative = self._strip_think_tags(result.get("content", ""))
        except Exception:  # noqa: BLE001
            pass  # Evaluator failure shouldn't block digest delivery

        # Step 3: Generate audio via TTS
        # Strip any markdown that slipped through (##, *, -, etc.)
        import re

        tts_text = re.sub(r"^#{1,6}\s+", "", narrative, flags=re.MULTILINE)
        tts_text = re.sub(r"^\s*[-*•]\s+", "", tts_text, flags=re.MULTILINE)
        tts_text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", tts_text)
        tts_text = tts_text.strip()

        tts_call = ToolCall(
            id="digest-tts-1",
            name="text_to_speech",
            arguments=json.dumps(
                {
                    "text": tts_text,
                    "voice_id": self._voice_id,
                    "backend": self._tts_backend,
                    "speed": self._voice_speed,
                }
            ),
        )
        tts_result = self._executor.execute(tts_call)
        audio_path = (
            tts_result.metadata.get("audio_path", "") if tts_result.success else ""
        )

        # Step 4: Store the artifact
        artifact = DigestArtifact(
            text=narrative,
            audio_path=Path(audio_path) if audio_path else Path(""),
            sections={},
            sources_used=sources,
            generated_at=datetime.now(),
            model_used=self._model,
            voice_used=self._voice_id,
            quality_score=quality_score,
            evaluator_feedback=evaluator_feedback,
        )

        store = DigestStore(db_path=self._digest_store_path)
        store.save(artifact)
        store.close()

        self._emit_turn_end(turns=1)
        return AgentResult(
            content=narrative,
            tool_results=[collect_result, tts_result],
            turns=1,
            metadata={
                "audio_path": audio_path,
                "sources_used": sources,
            },
        )
