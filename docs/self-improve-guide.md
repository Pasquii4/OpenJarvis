# Guía de Self-Improvement y Tracker Arquitectural en JARVIS

Esta guía describe los mecanismos automatizados de OpenJarvis para auto-entrenarse usando trazas de producción y monitorizar su salud y capacidades.

## 1. El Pipeline de Entrenamiento (Nightly Train)

El sistema incluye un script nocturno que se ejecuta cada noche para evaluar si JARVIS puede mejorar su desempeño usando ejemplos exitosos pasados.

### ¿Cómo funciona el LoRA SFT Local?
1. El job `nightly_train_job` (registrado en el `[scheduler]` como `0 3 * * *`) se lanza en background.
2. Usando `trace_to_dataset.py`, extrae las trazas recientes con `outcome = success` o feedback favorable.
3. **Privacidad**: Filtra de manera estricta los canales listados en `learning.data.exclude_channels` (por defecto, desactiva extracciones de `email` y `calendar`).
4. **Deduplicación**: Usa similitud pura de Levenshtein (>0.85) para ignorar preguntas casi idénticas y prevenir el overfitting.
5. Invoca el `LearningOrchestrator` de JARVIS.

### Simulación Dry-Run (Sin GPU en Windows)
Para el desarrollo en local bajo Windows sin CUDA o recursos de servidor, JARVIS se configura por defecto usando `dry_run = true` en `config.toml` -> `[learning.intelligence.sft]`.
En este modo:
- **No se dispara PyTorch/Transformers.**
- Se **simula un incremento del score** y se mockea un `adapter_path`.
- Se registra de manera transparente en Slack/Telegram una mejora y un resumen de las trazas recolectadas.

## 2. Panel de Salud de JARVIS

JARVIS extrae ahora telemetría agregada nativamente sobre latencias, retención de contextos y ratios de error, procesándolas eficientemente desde SQLite.

### Análisis con CLI
Puedes inspeccionar la salud del sistema usando:
```bash
jarvis health --days 7
jarvis health --format json
```
Esto retornará una rápida representación tabular semaforizada (RAG).

### Digest Semanal Asistido
Añadimos `health` de manera global al `[digest]` del sistema. Además, todos los **Domingos**, el sistema concatena al `health_digest` un chequeo del *Architect Status Tracker*. 

## 3. Tracker de Objetivos (Architect Goals)

EL archivo `configs/architect_goals.yaml` define la frontera de capacidades.
El comando `jarvis architect status` verifica la disponibilidad de conectores o `skills` mapeadas (Ej. ¿Soporta leer PDFs JARVIS? -> Si el registry contiene `file_read`).

Para evaluar las metas:
```bash
jarvis architect status
# Para formato tabla en reportes:
jarvis architect status --format markdown
```

**Adición de Nuevos Objetivos**: 
Cuando agregues nuevas integraciones como `discord_bot` o `control_lights`, actualiza `architect_goals.yaml` con una métrica descriptiva y expande `openjarvis.cli.architect_cmd.py` especificando el criteria que lo satisface (ej. importabilidad de `openjarvis.connectors.discord`). 
En CI (`JARVIS_CI=1`), un objetivo no satisfecho se marca como `SKIP`, pero en local evalúa como rojo (Falta).
