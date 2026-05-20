# PAI — Personal AI Assistant

A voice-enabled Personal AI Assistant that runs locally, controls your computer through natural language, and supports multi-turn conversations with LLM providers.

## Features

### Core
- Multi-turn conversational interface with context memory
- Google Gemini integration (cloud-based)
- Ollama integration (local, offline, no rate limits)
- Configurable via environment variables
- Structured logging with file rotation
- Clean error handling with custom exception hierarchy

### Voice (Phase 2)
- Wake word detection ("Hey Jarvis")
- Voice Activity Detection (Silero VAD)
- Speech-to-Text (Faster Whisper)
- Hands-free voice interaction loop

### Tools (Phase 3)
- **File System** — create, read, rename, delete, move, copy, list, search, open files, create folders
- **App Launcher** — launch any installed app (Steam, Discord, ChatGPT, VS Code, etc.), close apps, check running status
- **Terminal** — execute shell commands with safety denylist (disabled by default)
- Agentic tool-calling loop (LLM decides which tools to use)
- UWP/Windows Store app support (ChatGPT, WhatsApp, etc.)
- Windows Registry + AppData path resolution for installed apps

## Setup

### Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.ai/) installed locally for offline use
- (Optional) Microphone for voice mode

### Installation

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd PAI
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   venv\Scripts\activate     # Windows
   source venv/bin/activate  # Linux/macOS
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy the example environment file and configure it:

   ```bash
   cp .env.example .env
   ```

5. Edit `.env` with your settings:
   - Set `LLM_PROVIDER` to `gemini` or `ollama`
   - If using Gemini, add your `GEMINI_API_KEY`
   - If using Ollama, ensure `OLLAMA_HOST` and `OLLAMA_MODEL` are correct

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
You: open steam
PAI: The Steam application should now be open on your computer.

You: create a file on the Desktop named notes.txt with "Hello World"
PAI: Created the file notes.txt on your Desktop.

You: what files are on my Desktop?
PAI: Here are the files on your Desktop: ...

You: open discord
PAI: The Discord application should now be open on your computer.

You: what python version do I have?
PAI: You have Python 3.13.5 installed.
```

## Project Structure

```
PAI/
├── main.py              # CLI entry point (text + voice modes)
├── config.py            # Configuration system (env vars)
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── core/
│   ├── agent.py         # Agentic loop (tool calling + conversation)
│   ├── logger.py        # Loguru-based logging
│   └── exceptions.py    # Custom exception hierarchy
├── providers/
│   ├── base.py          # LLMProvider ABC, Message, ToolCall dataclasses
│   ├── gemini.py        # Google Gemini (native function calling)
│   └── ollama.py        # Ollama (local, tool calling via /api/chat)
├── tools/
│   ├── __init__.py      # Tool registry (get_tools, dispatch)
│   ├── base.py          # BaseTool ABC, ToolResult
│   ├── file_system.py   # File operations (10 operations)
│   ├── app_launcher.py  # App launch/close/status
│   └── terminal.py      # Shell command execution (disabled by default)
├── voice/
│   ├── pipeline.py      # Voice orchestration (wake → listen → transcribe)
│   ├── wake_word.py     # OpenWakeWord detector
│   ├── vad.py           # Silero Voice Activity Detection
│   ├── listener.py      # Utterance capture
│   ├── stt.py           # Faster Whisper transcription
│   └── mic.py           # Microphone capture
├── utils/
│   └── platform_utils.py # Cross-platform helpers, app resolution
├── memory/              # Memory/persistence (future)
├── tests/               # Test suite (54 tests)
└── logs/                # Runtime logs
```

## Testing

```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest tests/tools/       # Tool tests only
pytest tests/test_agent_tools.py  # Agent loop tests
```

## Configuration

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | Active provider (`gemini` or `ollama`) |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `AGENT_NAME` | `PAI` | Assistant display name |
| `MAX_SESSION_TURNS` | `20` | Max conversation turns before trimming |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### Tools

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_FILE_OPS` | `true` | Enable file system tool |
| `ENABLE_APP_LAUNCHER` | `true` | Enable app launcher tool |
| `ENABLE_TERMINAL` | `false` | Enable shell commands (security risk) |
| `MAX_TOOL_ITERATIONS` | `5` | Max tool calls per user message |

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

### Platform

| Variable | Default | Description |
|----------|---------|-------------|
| `SCREEN_SCALE_FACTOR` | `1.0` | HiDPI multiplier (future use) |

## Supported Apps (Launcher)

The app launcher can open any installed application. It resolves apps through:

1. **Alias lookup** — common names mapped to executables (40+ entries)
2. **Windows Registry** — App Paths key (Steam, Spotify, etc.)
3. **Program Files scan** — searches install directories
4. **AppData\Local** — Discord, Postman, etc.
5. **UWP/Store apps** — ChatGPT, WhatsApp via shell activation

## Safety

- Terminal tool is **disabled by default** — set `ENABLE_TERMINAL=true` to enable
- Dangerous commands are blocked (rm -rf, format, shutdown, etc.)
- File operations are sandboxed to user directories
- Tool iterations are capped at 5 per request

## License

MIT
