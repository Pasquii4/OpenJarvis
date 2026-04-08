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

echo.
echo     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
echo     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
echo     ██║███████║██████╔╝██║   ██║██║███████╗
echo ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
echo ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
echo  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
echo.
echo  Starting JARVIS...
echo.

REM Start the scheduler in the background
start /B uv run python -m openjarvis.agents.scheduler --config configs/jarvis_schedule.yaml

REM Start the chat REPL (default)
cd /d "%~dp0"
uv run jarvis chat

pause
