# JARVIS — Setup Guide for Windows

Get JARVIS running on your Windows PC with Groq Cloud API in 5 steps.

## Prerequisites

- **Windows 10/11**
- **Python 3.11+** — [Download](https://www.python.org/downloads/)
- **uv** (Python package manager) — Install: `pip install uv`
- **Groq API Key** (free) — [Get one](https://console.groq.com)

---

## Step 1: Clone and Enter the Repository

```powershell
git clone https://github.com/Pasquii4/OpenJarvis.git
cd OpenJarvis
```

---

## Step 2: Configure Environment Variables

```powershell
copy .env.example .env
```

Edit `.env` and fill in your keys:

| Variable | Required | How to Get |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | [console.groq.com](https://console.groq.com) → API Keys |
| `TELEGRAM_BOT_TOKEN` | Optional | Talk to [@BotFather](https://t.me/BotFather) on Telegram → `/newbot` |
| `TELEGRAM_USER_ID` | Optional | Talk to [@userinfobot](https://t.me/userinfobot) on Telegram |
| `TELEGRAM_CHAT_ID` | Optional | Same as User ID for private chats |
| `GOOGLE_CLIENT_ID` | Optional | [Google Cloud Console](https://console.cloud.google.com) → OAuth2 |
| `GOOGLE_CLIENT_SECRET` | Optional | Same as above |
| `GITHUB_TOKEN` | Optional | [GitHub Settings](https://github.com/settings/tokens) → Personal Access Token |

---

## Step 3: Install Dependencies

```powershell
uv sync
```

This installs all Python dependencies using the lockfile. Takes ~60 seconds on first run.

---

## Step 4: Launch JARVIS

```powershell
.\start_jarvis.bat
```

You'll see a menu:

```
[1] Chat interactivo (jarvis chat)
[2] Solo scheduler en background
[3] Servidor API (jarvis serve)
[4] Morning digest ahora (jarvis digest --fresh)
[5] Salir
```

Choose **1** for interactive chat. The config will be auto-copied to `%USERPROFILE%\.openjarvis\` on first run.

---

## Step 5: Verify It Works

### Quick test:

```powershell
uv run jarvis ask "Hola JARVIS, ¿qué puedes hacer?"
```

Expected: Response in Spanish from Groq in under 5 seconds.

### Memory test:

```powershell
uv run jarvis memory search "JARVIS"
```

### Config check:

```powershell
uv run jarvis config show
```

---

## Optional: Configure Telegram Bot

1. Talk to [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow the prompts
3. Copy the bot token to `TELEGRAM_BOT_TOKEN` in `.env`
4. Talk to [@userinfobot](https://t.me/userinfobot) to get your User ID
5. Set `TELEGRAM_USER_ID` and `TELEGRAM_CHAT_ID` in `.env`
6. Start a chat with your new bot and send any message
7. Restart JARVIS — the bot will now respond to your messages

### Scheduled Jobs (via Telegram)

JARVIS has 3 pre-configured jobs in `configs/jarvis_schedule.yaml`:

| Job | Schedule | Description |
|---|---|---|
| `morning_digest` | Daily at 8:00 AM | Morning briefing with email, calendar, news |
| `github_weekly` | Monday at 9:00 AM | GitHub repo summary for the week |
| `weekly_review` | Friday at 6:00 PM | Executive weekly review using memory |

All times are in **Europe/Madrid** timezone.

---

## Troubleshooting

### "GROQ_API_KEY not set"
→ Make sure `.env` exists and has your key. Run `copy .env.example .env` if needed.

### "openai package not installed"
→ Run `uv sync` to install all dependencies.

### Telegram bot not responding
→ Check that `TELEGRAM_BOT_TOKEN` and `TELEGRAM_USER_ID` are set correctly in `.env`.
→ Make sure you've started a conversation with your bot first.

### Memory search returns nothing
→ Memory populates as you use JARVIS. Send a few messages first.

### Config not loading
→ JARVIS looks for config in: `%USERPROFILE%\.openjarvis\config.toml` → `configs/openjarvis/config.toml`
→ `start_jarvis.bat` auto-copies on first run.
