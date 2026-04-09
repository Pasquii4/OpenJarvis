"""Router for system health and resource metrics.

Exposes real-time hardware telemetry so the Telegram channel and the
frontend dashboard can display CPU load, memory usage and (on Raspberry Pi)
the SoC temperature without importing psutil into the main codebase.
"""

from __future__ import annotations

import logging
import platform
import subprocess
import time
from typing import Optional

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])

# Record server start time so we can compute uptime
_SERVER_START: float = time.time()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_cpu_temperature() -> float:
    """Read the SoC temperature from a Raspberry Pi via ``vcgencmd``.

    On non-Pi platforms (Windows, macOS, generic Linux) this returns
    ``0.0`` silently so development machines are not affected.

    Returns
    -------
    float
        Temperature in degrees Celsius, or ``0.0`` on failure.
    """
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        # Output format: "temp=42.8'C"
        raw = result.stdout.strip()
        # Extract numeric part between "=" and "'"
        if "=" in raw and "'" in raw:
            temp_str = raw.split("=")[1].split("'")[0]
            return float(temp_str)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        # vcgencmd not available (development machine) — silently return 0
        pass
    except Exception:
        logger.debug("get_cpu_temperature: unexpected error", exc_info=True)
    return 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status")
async def system_status() -> dict:
    """Return current system resource utilisation.

    Uses ``psutil`` for cross-platform CPU/RAM metrics and attempts to
    read the Raspberry Pi temperature via ``vcgencmd`` (returns 0.0 on
    other platforms).

    Response fields:
    - ``cpu_percent``   — CPU usage (0–100 %)
    - ``ram_used_gb``   — RAM currently in use (GB, 2 dp)
    - ``ram_total_gb``  — Total installed RAM (GB, 1 dp)
    - ``ram_percent``   — RAM usage (0–100 %)
    - ``temperature``   — SoC temperature in °C (0.0 on non-Pi)
    - ``uptime_seconds``— Seconds since the server started
    - ``platform``      — Operating system string
    """
    try:
        import psutil  # optional but declared in dependencies
    except ImportError:
        logger.warning("psutil not installed — returning stub system status")
        return {
            "cpu_percent": 0.0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 0.0,
            "ram_percent": 0.0,
            "temperature": 0.0,
            "uptime_seconds": int(time.time() - _SERVER_START),
            "platform": platform.system(),
        }

    try:
        cpu_pct: float = psutil.cpu_percent(interval=0.1)
    except Exception:
        cpu_pct = 0.0

    try:
        vm = psutil.virtual_memory()
        ram_used_gb = round(vm.used / (1024 ** 3), 2)
        ram_total_gb = round(vm.total / (1024 ** 3), 1)
        ram_percent = round(vm.percent, 1)
    except Exception:
        ram_used_gb = ram_total_gb = ram_percent = 0.0

    temperature = get_cpu_temperature()
    uptime_seconds = int(time.time() - _SERVER_START)
    os_platform = platform.system()

    logger.debug(
        "system_status | cpu=%.1f%% ram=%.2f/%.1f GB temp=%.1f°C uptime=%ds",
        cpu_pct, ram_used_gb, ram_total_gb, temperature, uptime_seconds,
    )

    return {
        "cpu_percent": cpu_pct,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "ram_percent": ram_percent,
        "temperature": temperature,
        "uptime_seconds": uptime_seconds,
        "platform": os_platform,
    }


__all__ = ["router", "get_cpu_temperature"]
