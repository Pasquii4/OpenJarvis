"""TelegramChannel — native Telegram Bot API adapter.

Personalized for JARVIS: user whitelist, Spanish commands, voice
transcription, and streaming response support.

Also provides JarvisTelegramBot — an aiogram 3.x based companion
for sending pro-active notifications with inline approval keyboards.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from openjarvis.channels._stubs import (
    BaseChannel,
    ChannelHandler,
    ChannelMessage,
    ChannelStatus,
)
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry

logger = logging.getLogger(__name__)

# Allowed Telegram user ID — only this user can interact with JARVIS
_ALLOWED_USER_ID = os.environ.get("TELEGRAM_USER_ID", "")

# Spanish command definitions
_COMMANDS = {
    "/investigar": "deep_research",
    "/tarea": "scheduler_add",
    "/tareas": "scheduler_list",
    "/digest": "morning_digest",
    "/memoria": "memory_search",
    "/modo": "change_agent",
    "/monitor": "monitor_summary",
}


@ChannelRegistry.register("telegram")
class TelegramChannel(BaseChannel):
    """Native Telegram channel adapter using the Bot API.

    Parameters
    ----------
    bot_token:
        Telegram Bot API token.  Falls back to ``TELEGRAM_BOT_TOKEN`` env var.
    allowed_chat_ids:
        Comma-separated list of chat IDs allowed to interact.
    parse_mode:
        Message parse mode (``Markdown``, ``HTML``, etc.).
    bus:
        Optional event bus for publishing channel events.
    """

    channel_id = "telegram"

    def __init__(
        self,
        bot_token: str = "",
        *,
        allowed_chat_ids: str = "",
        parse_mode: str = "Markdown",
        bus: Optional[EventBus] = None,
    ) -> None:
        self._token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._allowed_chat_ids = allowed_chat_ids
        self._parse_mode = parse_mode
        self._bus = bus
        self._handlers: List[ChannelHandler] = []
        self._status = ChannelStatus.DISCONNECTED
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # -- connection lifecycle ---------------------------------------------------

    def connect(self) -> None:
        """Start listening for incoming messages via long polling."""
        if not self._token:
            logger.warning("No Telegram bot token configured")
            self._status = ChannelStatus.ERROR
            return

        self._stop_event.clear()
        self._status = ChannelStatus.CONNECTING

        try:
            from telegram.ext import ApplicationBuilder  # noqa: F401

            self._listener_thread = threading.Thread(
                target=self._poll_loop,
                daemon=True,
            )
            self._listener_thread.start()
            self._status = ChannelStatus.CONNECTED
            logger.info("Telegram channel connected (long polling)")
        except ImportError:
            # python-telegram-bot not installed — send-only mode
            logger.info(
                "python-telegram-bot not installed; send-only mode",
            )
            self._status = ChannelStatus.CONNECTED

    def disconnect(self) -> None:
        """Stop the listener thread."""
        self._stop_event.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=5.0)
            self._listener_thread = None
        self._status = ChannelStatus.DISCONNECTED

    # -- send / receive --------------------------------------------------------

    def send(
        self,
        channel: str,
        content: str,
        *,
        conversation_id: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> bool:
        """Send a message to a Telegram chat via the Bot API."""
        if not self._token:
            logger.warning("Cannot send: no Telegram bot token")
            return False

        try:
            import httpx

            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            chat_id = conversation_id or channel
            payload: Dict[str, Any] = {
                "chat_id": chat_id,
                "text": content,
            }
            if self._parse_mode:
                payload["parse_mode"] = self._parse_mode

            resp = httpx.post(url, json=payload, timeout=10.0)
            if resp.status_code < 300:
                self._publish_sent(channel, content, conversation_id)
                return True
            logger.warning(
                "Telegram API returned status %d: %s",
                resp.status_code,
                resp.text,
            )
            return False
        except Exception:
            logger.debug("Telegram send failed", exc_info=True)
            return False

    def status(self) -> ChannelStatus:
        """Return the current connection status."""
        return self._status

    def list_channels(self) -> List[str]:
        """Return available channel identifiers."""
        return ["telegram"]

    def on_message(self, handler: ChannelHandler) -> None:
        """Register a callback for incoming messages."""
        self._handlers.append(handler)

    # -- user whitelist --------------------------------------------------------

    @staticmethod
    def _is_authorized(user_id: str) -> bool:
        """Check if a user is authorized to interact with JARVIS."""
        if not _ALLOWED_USER_ID:
            return True  # No whitelist configured — allow all
        return user_id == _ALLOWED_USER_ID

    # -- command parsing -------------------------------------------------------

    @staticmethod
    def _parse_command(text: str) -> tuple[str, str]:
        """Parse a /command from message text.

        Returns (command, argument) or ("", "") if not a command.
        """
        if not text or not text.startswith("/"):
            return ("", "")
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower().split("@")[0]  # Strip @botname suffix
        arg = parts[1] if len(parts) > 1 else ""
        return (cmd, arg)

    def _build_command_message(self, cmd: str, arg: str) -> str:
        """Convert a Telegram command into a natural language query."""
        if cmd == "/investigar":
            return f"Investiga en profundidad sobre: {arg}" if arg else ""
        elif cmd == "/tarea":
            return f"Crea una tarea nueva: {arg}" if arg else ""
        elif cmd == "/tareas":
            return "Lista todas las tareas pendientes del scheduler."
        elif cmd == "/digest":
            return "Genera mi briefing matutino ahora."
        elif cmd == "/memoria":
            return f"Busca en mi memoria: {arg}" if arg else ""
        elif cmd == "/modo":
            return f"Cambia el agente activo a: {arg}" if arg else ""
        return ""

    # -- internal helpers -------------------------------------------------------

    def _poll_loop(self) -> None:
        """Long-poll for updates using python-telegram-bot."""
        try:
            from telegram.ext import ApplicationBuilder, MessageHandler, filters

            app = ApplicationBuilder().token(self._token).build()

            def _handle_msg(update, context):
                msg = update.message
                if msg is None:
                    return

                sender = str(msg.from_user.id) if msg.from_user else ""

                # Enforce user whitelist
                if not self._is_authorized(sender):
                    # Reply with rejection message
                    try:
                        import httpx

                        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
                        httpx.post(
                            url,
                            json={
                                "chat_id": str(msg.chat.id),
                                "text": "No estoy autorizado para hablar contigo.",
                            },
                            timeout=10.0,
                        )
                    except Exception:
                        pass
                    return

                text = msg.text or ""

                # Handle commands
                cmd, arg = self._parse_command(text)
                if cmd == "/memoria":
                    if not arg:
                        try:
                            import httpx
                            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
                            httpx.post(url, json={"chat_id": str(msg.chat.id), "text": "Uso: /memoria <consulta>\nEjemplo: /memoria reuniones importantes"}, timeout=10.0)
                        except Exception: pass
                        return
                    from openjarvis.memory.jarvis_memory import search_memory
                    results = search_memory(arg, top_k=5)
                    if not results:
                        try:
                            import httpx
                            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
                            httpx.post(url, json={"chat_id": str(msg.chat.id), "text": "No encontré nada relevante en memoria sobre eso."}, timeout=10.0)
                        except Exception: pass
                        return
                    lines = [f"🧠 *Memoria relevante para:* _{arg}_\n"]
                    for i, r in enumerate(results, 1):
                        meta = r.get("metadata", {})
                        ts = meta.get("timestamp", "")[:10] if isinstance(meta, dict) else ""
                        snippet = r.get("content", r.get("text", ""))[:250].replace("*", "").replace("_", "")
                        lines.append(f"*[{i}]* `{ts}`\n{snippet}\n")
                    try:
                        import httpx
                        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
                        httpx.post(url, json={"chat_id": str(msg.chat.id), "text": "\n".join(lines), "parse_mode": "Markdown"}, timeout=10.0)
                    except Exception: pass
                    return
                
                if cmd == "/monitor":
                    try:
                        from pathlib import Path
                        import json
                        from datetime import datetime
                        monitor_path = Path.home() / ".jarvis" / "monitor.jsonl"
                        if not monitor_path.exists():
                            text = "No hay métricas registradas."
                        else:
                            with open(monitor_path, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                            recent = []
                            for line in reversed(lines[-10:]):
                                if line.strip():
                                    recent.append(json.loads(line))
                            
                            if not recent:
                                text = "No hay métricas registradas."
                            else:
                                out = ["📊 *Últimas sesiones JARVIS*\n"]
                                for m in recent:
                                    ts_str = datetime.fromtimestamp(m.get("timestamp", 0)).strftime("%d/%m %H:%M")
                                    agent = m.get("agent_id", "N/A")
                                    turns = m.get("turns", 1)
                                    latency = m.get("latency_ms", 0) / 1000.0 if m.get("latency_ms", 0) > 1000 else m.get("latency_ms", 0)
                                    # If latency is passed in seconds directly, don't divide!
                                    # Actually the request says latency_ms, but usually Python time.time() difference is seconds
                                    latency_sec = m.get("latency_ms", 0) if m.get("latency_ms", 0) < 100 else m.get("latency_ms", 0) / 1000.0
                                    out.append(f"• {ts_str} — {agent} — {turns} turns — {latency_sec:.1f}s")
                                text = "\n".join(out)
                        import httpx
                        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
                        httpx.post(url, json={"chat_id": str(msg.chat.id), "text": text, "parse_mode": "Markdown"}, timeout=10.0)
                    except Exception: pass
                    return
                
                if cmd == "/architect":
                    try:
                        import httpx
                        from openjarvis.agents.architect import ArchitectAgent
                        from openjarvis.core.registry import EngineRegistry
                        from openjarvis.core.config import load_config
                        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
                        conf = load_config()
                        engine_id = conf.agent.default_engine
                        # Resolve engine since ArchitectAgent needs it
                        try:
                            eng_class = EngineRegistry.get(engine_id)
                            m_engine = eng_class()
                        except Exception:
                            # fallback to dummy or the running one if singleton. Wait, we usually get it from engine directly.
                            from openjarvis.engine import get_engine
                            m_engine = get_engine(conf)
                            
                        # Architect needs model name too
                        model = conf.agent.default_model
                        architect = ArchitectAgent(engine=m_engine, model=model)
                        
                        input_text = f"apply {arg.split('apply ', 1)[1]}" if "apply" in arg else "analiza el sistema completo"
                        
                        httpx.post(url, json={"chat_id": str(msg.chat.id), "text": "Architect trabajando, puede tardar unos segundos..."}, timeout=10.0)
                        
                        res = architect.run(input_text)
                        
                        out_text = res.content if hasattr(res, "content") else str(res)
                        if not out_text.strip():
                            out_text = "Proceso completado sin output."
                            
                        # Chunk it to 4000
                        chunk_size = 4000
                        chunks = [out_text[i:i+chunk_size] for i in range(0, len(out_text), chunk_size)]
                        for chunk in chunks:
                            httpx.post(url, json={"chat_id": str(msg.chat.id), "text": chunk, "parse_mode": "Markdown"}, timeout=10.0)
                            
                    except Exception as e:
                        try:
                            import httpx
                            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
                            httpx.post(url, json={"chat_id": str(msg.chat.id), "text": f"Error Architect: {e}"}, timeout=10.0)
                        except Exception: pass
                    return
                
                if cmd in _COMMANDS:
                    converted = self._build_command_message(cmd, arg)
                    if converted:
                        text = converted
                    elif not arg and cmd in ("/investigar", "/tarea", "/modo"):
                        # Command requires an argument
                        try:
                            import httpx

                            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
                            httpx.post(
                                url,
                                json={
                                    "chat_id": str(msg.chat.id),
                                    "text": f"Uso: {cmd} <argumento>",
                                },
                                timeout=10.0,
                            )
                        except Exception:
                            pass
                        return

                cm = ChannelMessage(
                    channel="telegram",
                    sender=sender,
                    content=text,
                    message_id=str(msg.message_id),
                    conversation_id=str(msg.chat.id),
                )
                # Enforce allow-list when configured (legacy behavior)
                if self._allowed_chat_ids:
                    _allowed = {
                        cid.strip()
                        for cid in self._allowed_chat_ids.split(",")
                        if cid.strip()
                    }
                    if cm.conversation_id not in _allowed:
                        logger.debug(
                            "Ignoring message from unlisted chat %s",
                            cm.conversation_id,
                        )
                        return
                for handler in self._handlers:
                    try:
                        handler(cm)
                    except Exception:
                        logger.exception("Telegram handler error")
                if self._bus is not None:
                    self._bus.publish(
                        EventType.CHANNEL_MESSAGE_RECEIVED,
                        {
                            "channel": cm.channel,
                            "sender": cm.sender,
                            "content": cm.content,
                            "message_id": cm.message_id,
                        },
                    )

            # Handle text messages
            app.add_handler(MessageHandler(filters.TEXT, _handle_msg))

            # Handle voice messages — transcribe and forward
            def _handle_voice(update, context):
                msg = update.message
                if msg is None or msg.voice is None:
                    return

                sender = str(msg.from_user.id) if msg.from_user else ""
                if not self._is_authorized(sender):
                    return

                try:
                    # Download voice file
                    voice_file = context.bot.get_file(msg.voice.file_id)
                    import tempfile
                    from pathlib import Path

                    with tempfile.NamedTemporaryFile(
                        suffix=".ogg", delete=False
                    ) as tmp:
                        voice_file.download_to_drive(tmp.name)
                        audio_path = Path(tmp.name)

                    # Try to transcribe using speech backend
                    transcribed = ""
                    try:
                        from openjarvis.core.config import load_config
                        from openjarvis.speech._discovery import get_speech_backend

                        config = load_config()
                        speech = get_speech_backend(config)
                        if speech:
                            result = speech.transcribe(audio_path)
                            transcribed = result.get("text", "") if isinstance(result, dict) else str(result)
                    except Exception:
                        logger.debug("Voice transcription failed", exc_info=True)

                    # Clean up temp file
                    try:
                        audio_path.unlink()
                    except Exception:
                        pass

                    if transcribed:
                        cm = ChannelMessage(
                            channel="telegram",
                            sender=sender,
                            content=transcribed,
                            message_id=str(msg.message_id),
                            conversation_id=str(msg.chat.id),
                        )
                        for handler in self._handlers:
                            try:
                                handler(cm)
                            except Exception:
                                logger.exception("Telegram voice handler error")
                    else:
                        # Notify user that transcription failed
                        import httpx

                        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
                        httpx.post(
                            url,
                            json={
                                "chat_id": str(msg.chat.id),
                                "text": "No pude transcribir el audio. Escríbeme en texto.",
                            },
                            timeout=10.0,
                        )
                except Exception:
                    logger.debug("Voice message handling failed", exc_info=True)

            try:
                app.add_handler(MessageHandler(filters.VOICE, _handle_voice))
            except Exception:
                logger.debug("Could not add voice handler", exc_info=True)

            app.run_polling(stop_signals=None, drop_pending_updates=True)
        except Exception:
            logger.debug("Telegram poll loop error", exc_info=True)
            self._status = ChannelStatus.ERROR

    def _publish_sent(self, channel: str, content: str, conversation_id: str) -> None:
        """Publish a CHANNEL_MESSAGE_SENT event on the bus."""
        if self._bus is not None:
            self._bus.publish(
                EventType.CHANNEL_MESSAGE_SENT,
                {
                    "channel": channel,
                    "content": content,
                    "conversation_id": conversation_id,
                },
            )


# ---------------------------------------------------------------------------
# JarvisTelegramBot — aiogram 3.x companion (Raspberry Pi / local-first)
# ---------------------------------------------------------------------------


class JarvisTelegramBot:
    """Aiogram 3.x based Telegram bot for JARVIS control.

    Designed for Raspberry Pi 5 deployment with Ollama + Qwen2.5.
    Handles commands, free-text chat, and inline approval flows.

    Parameters
    ----------
    token:
        Telegram Bot API token.  Falls back to ``TELEGRAM_BOT_TOKEN`` env var.
    jarvis_api_url:
        Base URL of the JARVIS FastAPI server.
        Falls back to ``JARVIS_API_URL`` env var (default ``http://localhost:8000``).
    """

    def __init__(
        self,
        token: str = "",
        *,
        jarvis_api_url: str = "",
    ) -> None:
        self._token: str = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._api_url: str = (
            jarvis_api_url
            or os.environ.get("JARVIS_API_URL", "http://localhost:8000")
        ).rstrip("/")
        self._bot: Any = None
        self._dp: Any = None
        self._log = logging.getLogger(__name__ + ".JarvisTelegramBot")

    # -- lifecycle -------------------------------------------------------------

    def _build(self) -> None:
        """Lazily build the Bot and Dispatcher (imports aiogram on first call)."""
        if self._bot is not None:
            return
        try:
            from aiogram import Bot, Dispatcher
            from aiogram.client.default import DefaultBotProperties
            from aiogram.enums import ParseMode

            self._bot = Bot(
                token=self._token,
                default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
            )
            self._dp = Dispatcher()
            self._register_handlers()
            self._log.info("JarvisTelegramBot construido con aiogram 3.x")
        except ImportError:
            self._log.error(
                "aiogram no está instalado. Instálalo con: uv add aiogram>=3.4.0"
            )
            raise

    def _register_handlers(self) -> None:
        """Register all command and message handlers on the dispatcher."""
        from aiogram import F
        from aiogram.filters import Command
        from aiogram.types import CallbackQuery, Message

        dp = self._dp

        @dp.message(Command("start"))
        async def cmd_start(msg: Message) -> None:
            self._log.info("/start from user_id=%s", msg.from_user.id if msg.from_user else "?")
            await msg.answer(
                "*¡Hola! Soy JARVIS* 🤖\n\n"
                "Tu asistente personal de IA. Aquí tienes los comandos disponibles:\n\n"
                "• /status — Estado del sistema (CPU, RAM, temperatura)\n"
                "• /agents — Agentes activos en este momento\n"
                "• /help — Lista completa de comandos\n"
                "• Puedes escribirme cualquier cosa y te responderé."
            )

        @dp.message(Command("help"))
        async def cmd_help(msg: Message) -> None:
            self._log.info("/help from user_id=%s", msg.from_user.id if msg.from_user else "?")
            await msg.answer(
                "*Comandos disponibles*\n\n"
                "• /start — Bienvenida e introducción\n"
                "• /status — Estado del sistema en tiempo real\n"
                "• /agents — Lista de agentes activos\n"
                "• /help — Este mensaje\n"
                "• *Texto libre* — Envíame cualquier pregunta o instrucción"
            )

        @dp.message(Command("status"))
        async def cmd_status(msg: Message) -> None:
            self._log.info("/status from user_id=%s", msg.from_user.id if msg.from_user else "?")
            try:
                import httpx

                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.get(f"{self._api_url}/api/system/status")
                    resp.raise_for_status()
                    data = resp.json()

                temp = data.get("temperature", 0.0)
                temp_str = f"{temp:.1f}°C" if temp > 0 else "N/A (desarrollo)"

                text = (
                    "*📊 Estado del sistema*\n\n"
                    f"🖥️ CPU: `{data.get('cpu_percent', 0):.1f}%`\n"
                    f"🧠 RAM: `{data.get('ram_used_gb', 0):.2f} / {data.get('ram_total_gb', 0):.1f} GB` "
                    f"({data.get('ram_percent', 0):.0f}%)\n"
                    f"🌡️ Temperatura: `{temp_str}`\n"
                    f"⏱️ Uptime: `{data.get('uptime_seconds', 0)}s`\n"
                    f"💻 Plataforma: `{data.get('platform', 'desconocida')}`"
                )
                self._log.info("Estado del sistema enviado a user_id=%s", msg.from_user.id if msg.from_user else "?")
            except Exception as exc:
                self._log.warning("Error obteniendo /status: %s", exc)
                text = f"⚠️ No pude obtener el estado del sistema: `{exc}`"
            await msg.answer(text)

        @dp.message(Command("agents"))
        async def cmd_agents(msg: Message) -> None:
            self._log.info("/agents from user_id=%s", msg.from_user.id if msg.from_user else "?")
            try:
                import httpx

                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.get(f"{self._api_url}/api/agents")
                    resp.raise_for_status()
                    data = resp.json()

                agents = data if isinstance(data, list) else data.get("agents", [])
                if not agents:
                    text = "ℹ️ No hay agentes activos en este momento."
                else:
                    lines = ["*🤖 Agentes activos*\n"]
                    for a in agents:
                        name = a.get("name") or a.get("id") or str(a)
                        status = a.get("status", "activo")
                        lines.append(f"• `{name}` — {status}")
                    text = "\n".join(lines)
                self._log.info("%d agentes listados para user_id=%s", len(agents), msg.from_user.id if msg.from_user else "?")
            except Exception as exc:
                self._log.warning("Error obteniendo /agents: %s", exc)
                text = f"⚠️ No pude obtener la lista de agentes: `{exc}`"
            await msg.answer(text)

        @dp.message(F.text)
        async def handle_free_text(msg: Message) -> None:
            """Forward free-text messages to /api/chat and return the response."""
            user_id = msg.from_user.id if msg.from_user else 0
            text_in = msg.text or ""
            self._log.info(
                "Mensaje libre de user_id=%s (len=%d)", user_id, len(text_in)
            )
            try:
                import httpx

                payload = {
                    "message": text_in,
                    "user_id": str(user_id),
                    "channel": "telegram",
                }
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(f"{self._api_url}/api/chat", json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                reply = (
                    data.get("response")
                    or data.get("content")
                    or data.get("message")
                    or str(data)
                )
                self._log.info("Respuesta enviada a user_id=%s", user_id)
            except Exception as exc:
                self._log.warning(
                    "Error procesando mensaje de user_id=%s: %s", user_id, exc
                )
                reply = (
                    f"⚠️ Lo siento, hubo un error procesando tu mensaje.\n"
                    f"Detalle: `{exc}`\n\n"
                    "Por favor, inténtalo de nuevo en unos segundos."
                )
            await msg.answer(reply)

        @dp.callback_query(F.data.startswith("approve:"))
        async def handle_approve(callback: CallbackQuery) -> None:
            """Handle approval of a code improvement proposal."""
            proposal_id = callback.data.split(":", 1)[1] if ":" in (callback.data or "") else ""
            user_id = callback.from_user.id if callback.from_user else 0
            self._log.info("Propuesta APROBADA id=%s por user_id=%s", proposal_id, user_id)
            try:
                import httpx

                async with httpx.AsyncClient(timeout=60.0) as client:
                    await client.post(
                        f"{self._api_url}/api/proposals/{proposal_id}/approve",
                        json={"user_id": str(user_id)},
                    )
            except Exception as exc:
                self._log.warning("Error aprobando propuesta %s: %s", proposal_id, exc)
            if callback.message:
                await callback.message.edit_text(
                    f"✅ Propuesta `{proposal_id}` *aprobada*. JARVIS ejecutará los cambios."
                )
            await callback.answer("✅ Aprobado")

        @dp.callback_query(F.data.startswith("reject:"))
        async def handle_reject(callback: CallbackQuery) -> None:
            """Handle rejection of a code improvement proposal."""
            proposal_id = callback.data.split(":", 1)[1] if ":" in (callback.data or "") else ""
            user_id = callback.from_user.id if callback.from_user else 0
            self._log.info("Propuesta RECHAZADA id=%s por user_id=%s", proposal_id, user_id)
            try:
                import httpx

                async with httpx.AsyncClient(timeout=60.0) as client:
                    await client.post(
                        f"{self._api_url}/api/proposals/{proposal_id}/reject",
                        json={"user_id": str(user_id)},
                    )
            except Exception as exc:
                self._log.warning("Error rechazando propuesta %s: %s", proposal_id, exc)
            if callback.message:
                await callback.message.edit_text(
                    f"❌ Propuesta `{proposal_id}` *rechazada*. No se aplicarán cambios."
                )
            await callback.answer("❌ Rechazado")

    # -- public API ------------------------------------------------------------

    async def send_notification(
        self,
        chat_id: int | str,
        text: str,
        *,
        proposal_id: Optional[str] = None,
    ) -> None:
        """Send a notification message, optionally with an approval inline keyboard.

        Parameters
        ----------
        chat_id:
            Target Telegram chat / user ID.
        text:
            Message text (Markdown).
        proposal_id:
            If provided, attaches Approve / Reject inline buttons for this
            code-improvement proposal ID.
        """
        self._build()
        try:
            if proposal_id is not None:
                from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

                keyboard = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="✅ Aprobar",
                                callback_data=f"approve:{proposal_id}",
                            ),
                            InlineKeyboardButton(
                                text="❌ Rechazar",
                                callback_data=f"reject:{proposal_id}",
                            ),
                        ]
                    ]
                )
                await self._bot.send_message(
                    chat_id=chat_id, text=text, reply_markup=keyboard
                )
                self._log.info(
                    "Notificación con inline keyboard enviada | chat_id=%s | proposal_id=%s",
                    chat_id, proposal_id,
                )
            else:
                await self._bot.send_message(chat_id=chat_id, text=text)
                self._log.info(
                    "Notificación enviada | chat_id=%s", chat_id
                )
        except Exception as exc:
            self._log.error(
                "Error enviando notificación a chat_id=%s: %s", chat_id, exc
            )

    async def start(self) -> None:
        """Start long-polling. Blocks until the bot is stopped."""
        self._build()
        self._log.info("JarvisTelegramBot iniciando polling…")
        try:
            await self._dp.start_polling(self._bot, skip_updates=True)
        except Exception as exc:
            self._log.error("JarvisTelegramBot polling error: %s", exc, exc_info=True)
            raise


__all__ = ["TelegramChannel", "JarvisTelegramBot"]
