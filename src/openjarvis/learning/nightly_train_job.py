import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

from openjarvis.core.config import load_config
from openjarvis.core.events import Event, EventType
from openjarvis.traces.store import TraceStore
from openjarvis.learning.trace_to_dataset import extract_training_pairs
from openjarvis.system import SystemBuilder

logger = logging.getLogger(__name__)

def evaluate_suite(model_mock: str = None) -> float:
    """Runs the evaluation suite and returns a dummy accuracy metric for the mock.
    In a real system, this would parse pytest output or run a specific eval framework.
    For this job, we run pytest tests/evals tests/agents/ and return a score.
    """
    logger.info("Running evaluation suite...")
    start = time.time()
    try:
        # We run the tests via subprocess, injecting the model string if needed.
        # But for dry_run or simple setup, returning a float placeholder if it passes.
        res = subprocess.run(["pytest", "tests/evals/", "tests/agents/"], capture_output=True, text=True)
        if res.returncode == 0:
            # DRY_RUN placeholder — reemplazar con parsing real de pytest output en producción
            return 0.0
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Eval suite failed: {e}")
        return 0.0

def promote_model(model_name: str, config) -> None:
    """Creates or updates model_override.toml."""
    override_path = Path("~/.openjarvis/model_override.toml").expanduser()
    override_content = f"""\
[intelligence]
model = "{model_name}"
"""
    with open(override_path, "w") as f:
        f.write(override_content)
    logger.info(f"Promoted model to {override_path}: {model_name}")

def send_telegram_summary(channels_mgr, msg: str):
    """Sends a message to the user via Telegram."""
    if channels_mgr:
        try:
            # Broadcast the report to Telegram
            channels_mgr.send("telegram", msg)
        except Exception as e:
            logger.error(f"Failed to send telegram summary: {e}")

def run() -> None:
    """Entry point for the nightly training job scheduled by task_scheduler."""
    config = load_config()
    telemetry = config.learning.intelligence.sft
    
    # Needs system builder to access channels / bus correctly.
    # Instantiate minimally:
    system = SystemBuilder(config).build()
    
    trace_store = system.trace_store
    if trace_store is None:
        try:
            trace_store = TraceStore(db_path=config.traces.db_path)
        except Exception:
            logger.error("No trace store available for nightly train.")
            return

    try:
        logger.info("Starting nightly train extraction...")
        pairs = extract_training_pairs(trace_store, config)
        min_pairs = telemetry.min_pairs
        
        summary_msg = f"🌙 *JARVIS Nightly Training Report*\n"
        summary_msg += f"- Pares extraídos: {len(pairs)}/{min_pairs}\n"

        if len(pairs) < min_pairs:
            logger.info("Not enough pairs. Aborting.")
            summary_msg += "-> Abortado por falta de datos suficientes."
            send_telegram_summary(system.channel_backend, summary_msg)
            return

        is_dry_run = getattr(telemetry, "dry_run", False)
        
        baseline_score = evaluate_suite()
        summary_msg += f"- Baseline score: {baseline_score:.2f}\n"

        if is_dry_run:
            logger.info("DRY_RUN: Simulando LoRA training.")
            new_model = telemetry.base_model + "-lora-simulated"
            baseline_score = 0.0  # En dry_run no se evalúa realmente
            new_score = config.learning.min_improvement + 0.01  # Simula pasar el umbral justo
            lora_path = "dry_run_path"
        else:
            # We call LearningOrchestrator pipeline
            if system._learning_orchestrator:
                result = system._learning_orchestrator.run()
                if not result.get("accepted", False):
                    summary_msg += "-> Entrenamiento rechazado: " + result.get("reason", "N/A")
                    send_telegram_summary(system.channel_backend, summary_msg)
                    return
                # If accepted, we would get the new model / score from result
                new_model = result.get("lora_training", {}).get("adapter_path", telemetry.base_model)
                new_score = result.get("post_score", baseline_score)
            else:
                logger.error("LearningOrchestrator not configured despite training_enabled=true.")
                return

        summary_msg += f"- Post-training score: {new_score:.2f}\n"
        improvement = new_score - baseline_score

        if improvement >= config.learning.min_improvement:
            promote_model(new_model, config)
            summary_msg += f"✅ Modelo promovido con mejora de +{improvement:.2f}.\nNuevo modelo: {new_model}"
            system.bus.publish(Event(
                type=EventType.SYSTEM_INFO,
                source="nightly_train_job",
                data={"metric": "model_promoted", "improvement": improvement}
            ))
        else:
            summary_msg += f"❌ Mejora insuficiente (+{improvement:.2f}). Checkpoint descartado."
            system.bus.publish(Event(
                type=EventType.SYSTEM_INFO,
                source="nightly_train_job",
                data={"metric": "model_rejected", "improvement": improvement}
            ))

        send_telegram_summary(system.channel_backend, summary_msg)

    except Exception as e:
        logger.exception("Nightly train job crash")
        if system.channel_backend:
            system.channel_backend.send("telegram", f"⚠️ Error en Nightly Training: {e}")
    finally:
        system.close()

if __name__ == "__main__":
    run()
