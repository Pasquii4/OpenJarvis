import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
from openjarvis.core.config import load_config
import os

def _get_model_version() -> str:
    # Check override
    override_path = Path("~/.openjarvis/model_override.toml").expanduser()
    if override_path.exists():
        try:
            import tomli
            with open(override_path, "rb") as f:
                data = tomli.load(f)
                return data.get("intelligence", {}).get("model", "unknown")
        except Exception:
            pass
    # Base config
    try:
        config = load_config()
        return config.intelligence.default_model
    except Exception:
        return "openrouter/deepseek/deepseek-r1"

def get_health_metrics(telemetry_db_path: str, traces_db_path: str, days: int = 7) -> Dict[str, Any]:
    cutoff_time = (datetime.now() - timedelta(days=days)).timestamp()
    
    metrics = {
        "avg_response_time_ms": 0.0,
        "total_tokens_per_day": {},
        "top_tools_used": [],
        "error_rate_by_type": {},
        "success_rate_by_channel": {},
        "sessions_per_day": {},
        "model_version": _get_model_version(),
        "error_rate": 0.0
    }

    if not os.path.exists(Path(traces_db_path).expanduser()):
        return metrics

    try:
        conn = sqlite3.connect(f"file:{Path(traces_db_path).expanduser()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Fetches for the last N days
        traces = cursor.execute("SELECT outcome, started_at, total_latency_seconds, total_tokens, metadata, result FROM traces WHERE started_at >= ?", (cutoff_time,)).fetchall()
        
        if not traces:
            return metrics

        total_latency = 0.0
        success_count = 0
        error_count = 0
        
        channel_stats = {}
        error_types = {}

        for row in traces:
            started = datetime.fromtimestamp(row["started_at"]).strftime("%Y-%m-%d")
            total_latency += row["total_latency_seconds"]

            # Tokens per day
            metrics["total_tokens_per_day"][started] = metrics["total_tokens_per_day"].get(started, 0) + row["total_tokens"]
            
            # Sessions approx by unique trace per day (simplified sessions equivalent)
            metrics["sessions_per_day"][started] = metrics["sessions_per_day"].get(started, 0) + 1

            raw_meta = row["metadata"] or "{}"
            meta = json.loads(raw_meta)
            channel = meta.get("channel", "unknown")
            if channel not in channel_stats:
                channel_stats[channel] = {"success": 0, "total": 0}
            
            channel_stats[channel]["total"] += 1

            if row["outcome"] == "success":
                success_count += 1
                channel_stats[channel]["success"] += 1
            else:
                error_count += 1
                etype = row["outcome"] or "unknown_error"
                # If outcome is literally just an error trace string
                if etype == "error":
                    # Attempt to parse result summary as specific error
                    res = row["result"]
                    if "context length" in res.lower():
                        etype = "context_limit"
                    elif "timeout" in res.lower():
                        etype = "timeout"
                    elif "rate limit" in res.lower():
                        etype = "rate_limit"

                error_types[etype] = error_types.get(etype, 0) + 1

        # Avg Response ms
        metrics["avg_response_time_ms"] = (total_latency / len(traces)) * 1000.0

        # Global error rate
        metrics["error_rate"] = float(error_count) / len(traces)

        # Error rate by type
        for etype, cnt in error_types.items():
            metrics["error_rate_by_type"][etype] = float(cnt) / len(traces)

        # Success rate by channel
        for ch, stats in channel_stats.items():
            metrics["success_rate_by_channel"][ch] = float(stats["success"]) / stats["total"]

        # Top tools
        tool_counts = {}
        # The step timestamp may not be totally accurate but it correlates with traces
        steps = cursor.execute("SELECT output, metadata, step_type FROM trace_steps WHERE step_type='tool' AND timestamp >= ?", (cutoff_time,)).fetchall()
        for s in steps:
            raw_smeta = s["metadata"] or "{}"
            smeta = json.loads(raw_smeta)
            tool_name = smeta.get("tool_name", "unknown")
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        sorted_tools = sorted(tool_counts.items(), key=lambda item: item[1], reverse=True)
        metrics["top_tools_used"] = dict(sorted_tools[:5])

        conn.close()

    except Exception as e:
        print(f"Error querying metrics: {e}")

    return metrics
