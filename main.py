"""
PAI — Personal Assistance and Intelligence

CLI entry point. Supports --voice flag for voice mode.
"""

import sys
import argparse

from core.logger import logger
from core.exceptions import ConfigError
import config


def main():
    """Main entry point for the PAI assistant."""
    parser = argparse.ArgumentParser(description="PAI — Personal AI Assistant")
    parser.add_argument(
        "--voice", action="store_true",
        help="Enable voice input mode (wake word → speak → response)"
    )
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  PAI — Personal Assistance and Intelligence")
    print(f"  Provider: {config.LLM_PROVIDER.upper()}")
    mode = "VOICE" if args.voice else "TEXT"
    print(f"  Mode: {mode}")
    print(f"{'='*50}\n")

    try:
        config.validate_config()
    except (ValueError, ConfigError) as e:
        print(f"\n[CONFIG ERROR] {e}\n")
        sys.exit(1)

    from providers import get_provider

    try:
        provider = get_provider()
    except Exception as e:
        print(f"\n[STARTUP ERROR] Could not load provider: {e}\n")
        sys.exit(1)

    # Health check — verify provider is reachable before starting chat loop
    try:
        provider.health_check()
        print(f"[OK] Provider {config.LLM_PROVIDER.upper()} is healthy.\n")
    except Exception as e:
        print(f"\n[PROVIDER ERROR] {e}\n")
        sys.exit(1)

    from core.agent import Agent

    agent = Agent(provider=provider)

    if args.voice:
        _run_voice_mode(agent)
    else:
        _run_text_mode(agent)


def _run_text_mode(agent):
    """Original text-based chat loop."""
    print("PAI is ready. Type your message and press Enter.")
    print("Commands: 'reset' to clear history | 'exit' to quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("[Conversation cleared]\n")
            continue

        response = agent.chat(user_input)
        print(f"\nPAI: {response}\n")


def _run_voice_mode(agent):
    """Voice input loop using wake word + STT."""
    from voice.pipeline import VoicePipeline
    from core.exceptions import AudioError

    print(f"Voice mode active. Say '{config.WAKE_WORD}' to speak.")
    print("Press Ctrl+C to quit.\n")

    try:
        pipeline = VoicePipeline()
        pipeline.load()
    except AudioError as e:
        print(f"\n[VOICE ERROR] {e}\n")
        sys.exit(1)

    def handle_transcription(text: str):
        print(f"\nYou: {text}")
        response = agent.chat(text)
        print(f"\nPAI: {response}\n")

    pipeline.run_forever(handle_transcription)


if __name__ == "__main__":
    main()
