# PAI — Personal AI Assistant

A voice-enabled Personal AI Assistant that runs locally, controls your computer through natural language, remembers your preferences across sessions, and supports multi-turn conversations with LLM providers.

## Features

### Phase 1 — Core Foundation
- Multi-turn conversational interface with context memory
- Google Gemini integration (cloud-based)
- Ollama integration (local, offline, no rate limits)
- Configurable via environment variables
- Structured logging with file rotation (Loguru)
- Clean error handling with custom exception hierarchy

### Phase 2 — Voice Input
- Wake word detection ("Hey Jarvis" via OpenWakeWord)
- Voice Activity Detection (Silero VAD)
- Speech-to-Text (Faster Whisper — tiny/base/small/medium/large)
- Hands-free voice interaction loop
- Configurable mic device, silence threshold, model size

### Phase 3 — Tool System
- **File System** — create, read, rename, delete, move, copy, list, search, open files, create folders
- **App Launcher** — launch any installed app (Steam, Discord, ChatGPT, VS Code, WhatsApp, etc.), close apps, check running status
- **Terminal** — execute PowerShell commands with safety denylist (disabled by default)
- Agentic tool-calling loop (LLM decides which tools to use, max 5 iterations)
- UWP/Windows Store app support (ChatGPT, WhatsApp via shell activation)
- Windows Registry + AppData + Program Files path resolution
- OneDrive Desktop path detection

### Phase 4 — Browser Control + Screen Vision
- **Browser Tool** — open URLs, Google search, YouTube search, scroll pages, read page title/URL
- **Screen Reader** — capture screenshot + send to vision model for description
- Chrome/Brave profile detection (opens with real login sessions)
- CDP (Chrome DevTools Protocol) for page info
- PyAutoGUI for typing and scrolling
- OS-level URL opening when browser already running
- Vision providers: Gemini (cloud) or Ollama (local/private)

### Phase 5 — Memory & Personalization
- **Long-term fact memory** — store key/value facts, persists across sessions in SQLite
- **Named shortcuts** — define custom command sequences (e.g. "work mode" → open VS Code and Spotify)
- **MemoryTool** — LLM-callable tool with 6 operations (remember/forget/list facts + define/delete/list shortcuts)
- Keyword-based retrieval — relevant facts auto-injected into system prompt each turn
- Memory stored at `~/.pai/memory.db` (configurable)

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
- Set `LLM_PROVIDER` to `gemini` or `ollama`
- If using Gemini: add your `GEMINI_API_KEY`
- If using Ollama: ensure `OLLAMA_HOST` and `OLLAMA_MODEL` are correct
- Enable tools as needed (`ENABLE_TERMINAL=true`, `BROWSER_ENABLED=true`, etc.)

## Running

### Text Mode
```bash
python main.py
```

### Voice Mode
```bash
python main.py --voice
```
Say "Hey Jarvis" to activate, then speak your command.

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

You: what do you know about me?
PAI: I know that your name is Sayan, your college is UEM, and your favorite editor is VS Code.

You: define a shortcut called "work mode" that opens VS Code and Spotify
PAI: Shortcut 'work mode' saved.

You: work mode
PAI: [opens VS Code and Spotify]

You: what python version do I have?
PAI: You have Python 3.13.5 installed.

You: list files on my desktop
PAI: [shows all files and folders]

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
│   ├── agent.py             # Agentic loop + memory injection
│   ├── logger.py            # Loguru-based logging
│   └── exceptions.py        # Custom exception hierarchy
├── providers/
│   ├── base.py              # LLMProvider ABC, Message, ToolCall dataclasses
│   ├── gemini.py            # Google Gemini (native function calling)
│   ├── ollama.py            # Ollama (local, tool calling via /api/chat)
│   └── vision_provider.py   # Vision model ABC + Gemini/Ollama implementations
├── tools/
│   ├── __init__.py          # Tool registry (get_tools, dispatch)
│   ├── base.py              # BaseTool ABC, ToolResult
│   ├── file_system.py       # File operations (10 operations)
│   ├── app_launcher.py      # App launch/close/status (40+ app aliases)
│   ├── terminal.py          # Shell command execution (PowerShell on Windows)
│   ├── browser.py           # Browser control (7 operations)
│   ├── screen_reader.py     # Screenshot → vision model → text
│   └── memory_tool.py       # Long-term memory CRUD (6 operations)
├── memory/
│   ├── __init__.py          # Module exports
│   ├── store.py             # SQLite CRUD for facts + shortcuts
│   ├── retriever.py         # Keyword extraction + context builder
│   └── shortcut_manager.py  # Shortcut validation wrapper
├── voice/
│   ├── pipeline.py          # Voice orchestration (wake → listen → transcribe)
│   ├── wake_word.py         # OpenWakeWord detector
│   ├── vad.py               # Silero Voice Activity Detection
│   ├── listener.py          # Utterance capture
│   ├── stt.py               # Faster Whisper transcription
│   └── mic.py               # Microphone capture
├── utils/
│   ├── platform_utils.py    # Cross-platform helpers, app/UWP resolution
│   ├── browser_utils.py     # Chrome/Brave profile detection, CDP helpers
│   └── vision_utils.py      # Screenshot capture (mss)
├── tests/                   # 105+ tests
│   ├── tools/               # Tool tests
│   ├── memory/              # Memory tests
│   └── ...
└── logs/                    # Runtime logs
```

## Testing

```bash
pytest                              # Run all tests
pytest -v                           # Verbose output
pytest tests/tools/                 # Tool tests only
pytest tests/memory/                # Memory tests only
pytest tests/test_agent_tools.py    # Agent loop tests
```

## Configuration Reference

### LLM Provider
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | Active provider (`gemini` or `ollama`) |
| `GEMINI_API_KEY` | — | Google Gemini API key |
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
| `TYPING_SPEED` | `0.04` | Seconds per character when typing |

### Vision
| Variable | Default | Description |
|----------|---------|-------------|
| `VISION_ENABLED` | `false` | Enable screen reader tool |
| `VISION_PROVIDER` | `gemini` | Vision backend (`gemini` or `ollama`) |
| `VISION_MODEL` | `gemma4:e4b` | Ollama vision model tag |
| `SCREENSHOT_DELAY` | `0.3` | Seconds to wait before capturing |

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
| `MIC_DEVICE` | — | Mic device index (empty = system default) |

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
- File operations use full absolute paths
- Tool iterations capped at 5 per request
- Browser/Vision tools disabled by default

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU VRAM | 4 GB (llama3.1:8b) | 10 GB (gemma4:e4b) |
| Disk | 10 GB (models) | 20 GB |
| Mic | Any USB/built-in | Low-latency headset |

## License

MIT
