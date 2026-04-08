@echo off
REM ============================================================
REM  JARVIS — Personal AI Assistant
REM  Start script for Windows
REM ============================================================

TITLE JARVIS Personal Assistant

REM Load environment variables from .env if present
IF EXIST "%~dp0.env" (
    for /f "usebackq tokens=1,2 delims==" %%a in ("%~dp0.env") do (
        REM Skip comments
        echo %%a | findstr /b "#" >nul || set "%%a=%%b"
    )
)

REM Check required env vars
IF "%GROQ_API_KEY%"=="" (
    echo [ERROR] GROQ_API_KEY not set. Copy .env.example to .env and fill in your key.
    pause
    exit /b 1
)

REM Optional: warn about Telegram
IF "%TELEGRAM_BOT_TOKEN%"=="" (
    echo [WARN] TELEGRAM_BOT_TOKEN not set. Scheduled jobs won't send Telegram messages.
    echo        Set it in .env to enable Telegram notifications.
    echo.
)

REM ── Auto-setup: create ~/.openjarvis and copy config ──
IF NOT EXIST "%USERPROFILE%\.openjarvis" (
    echo [SETUP] Creating %USERPROFILE%\.openjarvis\ ...
    mkdir "%USERPROFILE%\.openjarvis"
)

IF NOT EXIST "%USERPROFILE%\.openjarvis\config.toml" (
    IF EXIST "%~dp0configs\openjarvis\config.toml" (
        echo [SETUP] Copying config.toml to %USERPROFILE%\.openjarvis\ ...
        copy "%~dp0configs\openjarvis\config.toml" "%USERPROFILE%\.openjarvis\config.toml" >nul
    )
)

echo.
echo     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
echo     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
echo     ██║███████║██████╔╝██║   ██║██║███████╗
echo ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
echo ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
echo  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
echo.
echo  JARVIS Personal Assistant — Windows + Groq
echo  ──────────────────────────────────────────
echo.
echo  [1] Chat interactivo (jarvis chat)
echo  [2] Solo scheduler en background
echo  [3] Servidor API (jarvis serve)
echo  [4] Morning digest ahora (jarvis digest --fresh)
echo  [5] Salir
echo.

cd /d "%~dp0"

set /p choice="  Selecciona una opcion [1-5]: "

IF "%choice%"=="1" (
    echo.
    echo  Starting scheduler in background...
    start /B uv run python -m openjarvis.agents.scheduler --config configs/jarvis_schedule.yaml
    echo  Starting chat REPL...
    echo.
    uv run jarvis chat
) ELSE IF "%choice%"=="2" (
    echo.
    echo  Starting scheduler only...
    echo  Press Ctrl+C to stop.
    echo.
    uv run python -m openjarvis.agents.scheduler --config configs/jarvis_schedule.yaml
) ELSE IF "%choice%"=="3" (
    echo.
    echo  Starting API server on http://127.0.0.1:8000 ...
    echo  Press Ctrl+C to stop.
    echo.
    uv run jarvis serve
) ELSE IF "%choice%"=="4" (
    echo.
    echo  Generating morning digest...
    echo.
    uv run jarvis digest --fresh
) ELSE IF "%choice%"=="5" (
    exit /b 0
) ELSE (
    echo  Invalid option. Starting chat by default...
    echo.
    start /B uv run python -m openjarvis.agents.scheduler --config configs/jarvis_schedule.yaml
    uv run jarvis chat
)

pause
