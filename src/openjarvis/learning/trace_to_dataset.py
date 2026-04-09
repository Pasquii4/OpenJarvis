"""Extractor for LoRA SFT pairs from interaction traces."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from openjarvis.core.events import get_event_bus, Event, EventType

logger = logging.getLogger(__name__)

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculates the normalized similarity (1.0 = identical)."""
    if not s1 and not s2:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (dist / max_len)

def extract_training_pairs(trace_store: Any, config: Any) -> List[Dict[str, Any]]:
    """Extract fine-tuning pairs from traces, applying privacy and deduplication.
    
    Parameters
    ----------
    trace_store:
        The TraceStore instance to query from.
    config:
        JarvisConfig active instance.
        
    Returns
    -------
    List of dictionary pairs format {"instruction", "input", "output"}.
    """
    sft_config = config.learning.intelligence.sft
    data_config = config.learning.data
    
    min_quality = data_config.min_quality_score
    exclude_channels = data_config.exclude_channels
    privacy_filter = data_config.privacy_filter

    # Fetch max pairs requested times 2 to account for filter drops
    limit = sft_config.max_pairs * 2
    traces = trace_store.list_traces(limit=limit)

    pairs = []
    num_filtered_privacy = 0
    num_deduplicated = 0

    for trace in traces:
        # Check outcome or feedback threshold
        if trace.outcome != "success":
            if trace.feedback is None or trace.feedback < min_quality:
                continue

        # Privacy Filter
        if privacy_filter:
            channel = trace.metadata.get("channel", "")
            if channel in exclude_channels:
                num_filtered_privacy += 1
                continue

        instruction = "Eres JARVIS, el asistente personal inteligente de Pau."
        input_text = trace.query
        output_text = trace.result

        if not input_text or not output_text:
            continue

        pair = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
        
        # Levenshtein deduplication based on input text
        is_duplicate = False
        for existing in pairs:
            sim = normalized_levenshtein_similarity(input_text, existing["input"])
            if sim > 0.85:
                is_duplicate = True
                break
        
        if is_duplicate:
            num_deduplicated += 1
            continue

        pairs.append(pair)
        if len(pairs) >= sft_config.max_pairs:
            break

    # Save to JSONL dataset
    export_dir = Path(data_config.export_dir).expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    export_path = export_dir / f"dataset_{today}.jsonl"

    with open(export_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Record telemetry locally on the generalized bus using a raw struct
    bus = get_event_bus()
    bus.publish(Event(
        type=EventType.SYSTEM_INFO,
        source="trace_to_dataset",
        data={
            "telemetry_metric": "training_extraction",
            "num_pairs_extracted": len(pairs),
            "num_filtered_privacy": num_filtered_privacy,
            "num_deduplicated": num_deduplicated,
        }
    ))

    logger.info("Extracted %d pairs. Filtered privacy: %d. Deduplicated: %d.", len(pairs), num_filtered_privacy, num_deduplicated)
    return pairs
