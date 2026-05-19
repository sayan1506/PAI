# PAI — Personal AI Assistant

A terminal-based Personal AI Assistant that supports multi-turn conversations with LLM providers (Google Gemini and local Ollama).

## Features

- Multi-turn conversational interface with context memory
- Google Gemini integration (cloud-based, free tier)
- Ollama integration (local, offline capable)
- Configurable via environment variables
- Structured logging with file rotation
- Clean error handling with custom exception hierarchy

## Setup

### Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.ai/) installed locally for offline use

### Installation

1. Clone the repository and navigate to the project directory.

2. Create a virtual environment and activate it:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
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

### Running

```bash
python main.py
```

### Commands

- Type a message and press Enter to chat
- Type `reset` to clear conversation history
- Type `exit` or `quit` to leave
- Press `Ctrl+C` to exit at any time

## Project Structure

```
PAI/
├── main.py              # CLI entry point
├── config.py            # Configuration system
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── core/                # Core modules (agent, logger, exceptions)
├── providers/           # LLM provider implementations
├── tools/               # Tool integrations (future)
├── voice/               # Voice features (future)
├── memory/              # Memory/persistence (future)
├── utils/               # Shared utilities
└── tests/               # Test suite
```

## Testing

Run the test suite:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | Active LLM provider |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `AGENT_NAME` | `PAI` | Assistant display name |
| `MAX_SESSION_TURNS` | `20` | Max conversation turns before trimming |
