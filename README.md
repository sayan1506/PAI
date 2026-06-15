# PAI — Personal AI Assistant

A voice-enabled Personal AI Assistant that runs locally, controls your computer through natural language, remembers your preferences across sessions, speaks its responses aloud, and works with five different LLM providers — cloud or fully offline.

Talk to it by voice ("Hey Jarvis...") or type in the terminal. PAI uses an agentic tool-calling loop to actually do things on your machine: open apps, manage files, browse the web, read your screen, check the weather, set reminders, and remember facts about you.

## Features

### Core Foundation
- Multi-turn conversational interface with context memory
- Google Gemini integration (cloud-based)
- Ollama integration (local, offline, no rate limits)
- Configurable via environment variables
- Structured logging with file rotation (Loguru)
- Clean error handling with custom exception hierarchy

### Voice Input
- Wake word detection ("Hey Jarvis" via OpenWakeWord)
- Voice Activity Detection (Silero VAD)
- Speech-to-Text (Faster Whisper — tiny/base/small/medium/large)
- Hands-free voice interaction loop
- Configurable mic device, silence threshold, model size

### Tool System
- **File System** — create, read, rename, delete, move, copy, list, search, open files, create folders
- **App Launcher** — launch any installed app (Steam, Discord, ChatGPT, VS Code, WhatsApp, etc.), close apps, check running status
- **Terminal** — execute PowerShell commands with safety denylist (disabled by default)
- Agentic tool-calling loop (LLM decides which tools to use, max 5 iterations)
- UWP/Windows Store app support (ChatGPT, WhatsApp via shell activation)
- Windows Registry + AppData + Program Files path resolution
- OneDrive Desktop path detection

### Browser Control + Screen Vision
- **Browser Tool** — open URLs, web search, click links by text, scroll pages, read full page text
- **Screen Reader** — capture screenshot + send to vision model for description
- Chrome/Brave profile detection (opens with real login sessions)
- CDP (Chrome DevTools Protocol) for fast, accurate page reading
- PyAutoGUI for visible typing and scrolling
- Vision providers: Gemini (cloud) or Ollama (local/private)

### Memory & Personalization
- **Long-term fact memory** — store key/value facts, persists across sessions in SQLite
- **Named shortcuts** — define custom command sequences (e.g. "work mode" → open VS Code and Spotify)
- **MemoryTool** — LLM-callable tool with 6 operations (remember/forget/list facts + define/delete/list shortcuts)
- Keyword-based retrieval — relevant facts auto-injected into system prompt each turn
- Memory stored at `~/.pai/memory.db` (configurable)

### TTS Output & UI
- **Text-to-Speech** — Kokoro neural TTS (default) with pyttsx3 fallback
- Configurable voice, speed, and engine
- **Barge-in** — interrupt PAI mid-speech by speaking again
- **Conversation overlay** — small always-on-top window showing the latest exchange
- **Desktop notifications** — toast notifications for responses and reminders

### Enhanced Browser Control
- Click links and buttons by visible text label or x,y coordinates
- Read the full visible text of any page back to the LLM
- Configurable Chrome binary path, CDP timeout, and typing delay
- OS-level URL opening when the browser is already running

### Weather & Reminders
- **Weather Tool** — current conditions for any city via OpenWeatherMap (temp, humidity, wind), metric or imperial
- **Reminder Tool** — set, list, and cancel time-based reminders ("remind me in 10m", "remind me at 2:30pm")
- Reminders persist across restarts in SQLite and fire as desktop notifications
- Background scheduler polls for due reminders on a configurable interval

### Multi-Provider Support
Five swappable LLM backends behind one interface — switch with a single config value:
- **Gemini** — Google, native function calling
- **Ollama** — local/offline, tool calling via `/api/chat`
- **GitHub Models** — free tier via a GitHub PAT
- **Anthropic** — Claude models
- **OpenAI** — GPT models
- Automatic Gemini → Ollama fallback when Gemini hits a rate limit

### Hardening & Polish
- **Startup health check** — probes each component (provider, memory DB, mic, STT) and degrades gracefully (e.g. auto-switch to text mode if no mic)
- **Global hotkey** — trigger PAI with a keyboard combo (default `Ctrl+Shift+J`) as an alternative to the wake word
- **Rate tracking** — daily Gemini request counter with warning thresholds
- **Destructive-action confirmation** — Y/N gate before risky operations
- **Turn timing** — per-stage latency logging for each interaction

## Setup

### Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.ai/) installed locally for offline use
- (Optional) Microphone for voice mode
- (Optional) NVIDIA GPU for faster Ollama inference

### Installation

```bash
git clone <repo-url>
cd PAI
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your settings
```

### Configuration

Edit `.env`:
- Set `LLM_PROVIDER` to one of `gemini`, `ollama`, `github`, `anthropic`, or `openai`
- Add the matching API key for your chosen provider (Ollama needs none)
- Enable tools as needed (`ENABLE_TERMINAL=true`, `BROWSER_ENABLED=true`, `WEATHER_ENABLED=true`, etc.)
- Configure voice and TTS options if using voice mode

## Running

### Text Mode
```bash
python main.py
```

### Voice Mode
```bash
python main.py --voice
```
Say "Hey Jarvis" to activate, then speak your command. PAI speaks its response back. You can also press the global hotkey (`Ctrl+Shift+J`) to trigger it.

### Commands (Text Mode)
- Type a message and press Enter to chat
- `reset` — clear conversation history
- `exit` or `quit` — leave
- `Ctrl+C` — exit at any time

## Example Usage

```
You: remember that my name is Sayan
PAI: I have remembered that your name is Sayan.

You: open steam
PAI: The Steam application should now be open on your computer.

You: search on youtube for lo-fi music
PAI: Navigated to YouTube search results for "lo-fi music".

You: create a file called notes.txt on desktop with "Hello World"
PAI: Created file: C:\Users\sayan\OneDrive\Desktop\notes.txt

You: what's the weather in London?
PAI: It's currently 14°C in London with light rain, 82% humidity and a 4 m/s wind.

You: remind me in 20 minutes to check the oven
PAI: Reminder set for 20 minutes from now: "check the oven".

You: what do you know about me?
PAI: I know that your name is Sayan, your college is UEM, and your favorite editor is VS Code.

You: define a shortcut called "work mode" that opens VS Code and Spotify
PAI: Shortcut 'work mode' saved.

You: work mode
PAI: [opens VS Code and Spotify]

You: open the Plan folder in vs code
PAI: I opened the 'Plan' folder in VS Code.
```

## Project Structure

```
PAI/
├── main.py                  # CLI entry point (text + voice modes)
├── config.py                # Configuration system (env vars)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── core/
│   ├── agent.py             # Agentic loop, memory injection, provider fallback
│   ├── logger.py            # Loguru-based logging
│   ├── exceptions.py        # Custom exception hierarchy
│   ├── health.py            # Startup component health checks
│   ├── hotkey.py            # Global hotkey listener
│   ├── rate_tracker.py      # Daily request counting / limits
│   ├── reminder_scheduler.py# Background reminder polling + firing
│   ├── confirmation.py      # Destructive-action Y/N gate
│   └── timing.py            # Per-turn / per-stage latency timing
├── providers/
│   ├── base.py              # LLMProvider ABC, Message, ToolCall dataclasses
│   ├── gemini.py            # Google Gemini (native function calling)
│   ├── ollama.py            # Ollama (local, tool calling via /api/chat)
│   ├── github_models.py     # GitHub Models (free via PAT)
│   ├── anthropic.py         # Anthropic Claude
│   ├── openai_provider.py   # OpenAI GPT
│   ├── openai_compat.py     # Shared OpenAI-compatible client logic
│   └── vision_provider.py   # Vision model ABC + Gemini/Ollama implementations
├── tools/
│   ├── __init__.py          # Tool registry (get_tools, dispatch)
│   ├── base.py              # BaseTool ABC, ToolResult
│   ├── file_system.py       # File operations (10 operations)
│   ├── app_launcher.py      # App launch/close/status (40+ app aliases)
│   ├── terminal.py          # Shell command execution (PowerShell on Windows)
│   ├── browser.py           # Browser control (open/search/click/scroll/read/close)
│   ├── browser_launcher.py  # Chrome/Brave launch with real profile
│   ├── browser_cdp.py       # Chrome DevTools Protocol helpers
│   ├── screen_reader.py     # Screenshot → vision model → text
│   ├── memory_tool.py       # Long-term memory CRUD (6 operations)
│   ├── weather.py           # Current weather via OpenWeatherMap
│   └── reminder.py          # Set/list/cancel time-based reminders
├── memory/
│   ├── __init__.py          # Module exports
│   ├── store.py             # SQLite CRUD for facts, shortcuts, reminders
│   ├── retriever.py         # Keyword extraction + context builder
│   └── shortcut_manager.py  # Shortcut validation wrapper
├── voice/
│   ├── pipeline.py          # Voice orchestration (wake → listen → transcribe)
│   ├── wake_word.py         # OpenWakeWord detector
│   ├── vad.py               # Silero Voice Activity Detection
│   ├── listener.py          # Utterance capture
│   ├── stt.py               # Faster Whisper transcription
│   ├── mic.py               # Microphone capture
│   ├── tts.py               # Text-to-Speech (Kokoro / pyttsx3)
│   └── barge_in.py          # Interrupt TTS playback by speaking
├── ui/
│   ├── overlay.py           # Always-on-top conversation overlay (tkinter)
│   └── notifier.py          # Desktop toast notifications
├── utils/
│   ├── platform_utils.py    # Cross-platform helpers, app/UWP resolution
│   ├── browser_utils.py     # Chrome/Brave profile detection, CDP helpers
│   └── vision_utils.py      # Screenshot capture (mss)
├── tests/                   # Unit, integration, and property (Hypothesis) tests
│   ├── tools/               # Tool tests
│   ├── memory/              # Memory tests
│   ├── voice/               # Voice/TTS tests
│   ├── ui/                  # Overlay/notifier tests
│   ├── core/                # Core module tests
│   └── integration/         # End-to-end tests
└── logs/                    # Runtime logs
```

## Testing

```bash
pytest                              # Run all tests
pytest -v                           # Verbose output
pytest tests/tools/                 # Tool tests only
pytest tests/memory/                # Memory tests only
pytest tests/integration/           # Integration tests
```

## Configuration Reference

### LLM Provider
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | Active provider (`gemini`, `ollama`, `github`, `anthropic`, `openai`) |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GITHUB_TOKEN` | — | GitHub PAT with `models` scope |
| `GITHUB_MODEL` | `gpt-4o` | GitHub Models model name |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Anthropic model name |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `AGENT_NAME` | `PAI` | Assistant display name |
| `MAX_SESSION_TURNS` | `20` | Max conversation turns before trimming |

### Tools
| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_FILE_OPS` | `true` | Enable file system tool |
| `ENABLE_APP_LAUNCHER` | `true` | Enable app launcher tool |
| `ENABLE_TERMINAL` | `false` | Enable shell commands (security risk) |
| `MAX_TOOL_ITERATIONS` | `5` | Max tool calls per user message |

### Browser
| Variable | Default | Description |
|----------|---------|-------------|
| `BROWSER_ENABLED` | `false` | Enable browser control tool |
| `BROWSER_APP` | `chrome` | Browser to use (`chrome` or `brave`) |
| `BROWSER_REMOTE_PORT` | `9222` | CDP debugging port |
| `BROWSER_BINARY` | — | Override Chrome binary path (auto-detect if empty) |
| `BROWSER_CDP_TIMEOUT` | `10` | Seconds to wait for the debugging port |
| `TYPING_SPEED` | `0.04` | Seconds per character when typing |
| `BROWSER_TYPE_DELAY` | `0.04` | Seconds between keystrokes |

### Vision
| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_ENABLED` | `false` | Enable screen reader tool |
| `VISION_PROVIDER` | `gemini` | Vision backend (`gemini` or `ollama`) |
| `VISION_MODEL` | `gemma3:4b` | Ollama vision model tag |
| `SCREENSHOT_DELAY` | `0.3` | Seconds to wait before capturing |
| `SCREEN_SCALE_FACTOR` | `1.0` | Increase for HiDPI/4K screens |

### Memory
| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_ENABLED` | `true` | Enable long-term memory |
| `MEMORY_DB_PATH` | `~/.pai/memory.db` | SQLite database path |
| `MEMORY_INJECT_TOP_K` | `5` | Max facts injected per turn |

### Voice
| Variable | Default | Description |
|----------|---------|-------------|
| `WAKE_WORD` | `hey jarvis` | Wake word phrase |
| `WAKE_WORD_SENSITIVITY` | `0.5` | Detection threshold (lower = more sensitive) |
| `STT_MODEL` | `faster-whisper` | Speech-to-text engine |
| `STT_SIZE` | `base` | Whisper model size (tiny/base/small/medium/large) |
| `VOICE_ENABLED` | `true` | Enable voice subsystem |
| `VAD_SILENCE_MS` | `1200` | Silence duration to end utterance (ms) |
| `MIC_SAMPLE_RATE` | `16000` | Microphone sample rate |
| `MIC_CHANNELS` | `1` | Microphone channels |
| `MIC_DEVICE` | — | Mic device index (empty = system default) |

### TTS & Output
| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ENABLED` | `true` | Enable text-to-speech output |
| `TTS_ENGINE` | `kokoro` | TTS engine (`kokoro` or `pyttsx3`) |
| `TTS_SPEED` | `1.0` | Speech speed multiplier |
| `TTS_VOICE` | `af_heart` | Voice name for Kokoro TTS |
| `BARGE_IN_ENABLED` | `true` | Allow interrupting speech by speaking |
| `OVERLAY_ENABLED` | `true` | Show floating conversation overlay |
| `NOTIFICATIONS_ENABLED` | `true` | Show desktop toast notifications |

### Weather
| Variable | Default | Description |
|----------|---------|-------------|
| `WEATHER_ENABLED` | `false` | Enable the weather tool |
| `WEATHER_API_KEY` | — | OpenWeatherMap API key |
| `WEATHER_UNITS` | `metric` | `metric` (°C, m/s) or `imperial` (°F, mph) |
| `WEATHER_DEFAULT_LOCATION` | — | Fallback city when none is given |

### Reminders
| Variable | Default | Description |
|----------|---------|-------------|
| `REMINDERS_ENABLED` | `true` | Enable time-based reminders |
| `REMINDER_POLL_INTERVAL` | `10` | Seconds between reminder DB checks |

### Hardening
| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_HEALTH_CHECK` | `false` | Skip the startup component probe |
| `HOTKEY_ENABLED` | `true` | Enable global hotkey activation |
| `HOTKEY_COMBO` | `<ctrl>+<shift>+j` | Hotkey combo (pynput format) |
| `GEMINI_DAILY_LIMIT` | `1000` | Daily Gemini request ceiling |
| `GEMINI_WARN_AT` | `800` | First warning threshold |
| `CONFIRM_DESTRUCTIVE` | `true` | Require Y/N before destructive actions |

## Supported Apps (Launcher)

The app launcher resolves apps through:
1. **Alias lookup** — 40+ common names mapped to executables
2. **Windows Registry** — App Paths key (Steam, Spotify, etc.)
3. **Program Files scan** — searches install directories
4. **AppData\Local** — Discord, Postman, etc.
5. **UWP/Store apps** — ChatGPT, WhatsApp via shell activation

## Safety

- Terminal tool is **disabled by default** — set `ENABLE_TERMINAL=true` to enable
- Dangerous commands are blocked (rm -rf, format, shutdown, etc.)
- Destructive actions prompt for Y/N confirmation (`CONFIRM_DESTRUCTIVE=true`)
- File operations use full absolute paths
- Tool iterations capped at 5 per request
- Browser, Vision, and Weather tools are disabled by default

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU VRAM | 4 GB (llama3.1:8b) | 10 GB (gemma3:4b) |
| Disk | 10 GB (models) | 20 GB |
| Mic | Any USB/built-in | Low-latency headset |

## License

MIT
