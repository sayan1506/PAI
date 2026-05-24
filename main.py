"""
PAI — Personal Assistance and Intelligence

CLI entry point. Supports --voice flag for voice mode.
"""

import sys
import argparse

from core.logger import logger
from core.exceptions import ConfigError
from ui.overlay import ConversationOverlay
from ui.notifier import notify
from voice.tts import get_tts
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

    from core.health import run_health_check
    health = run_health_check()
    print(health.summary())
    print()
    if health.critical_failure:
        sys.exit(1)

    # Provider network ping (separate from local component check)
    try:
        provider.health_check()
        print(f"[OK] Provider {config.LLM_PROVIDER.upper()} reachable.\n")
    except Exception as e:
        print(f"\n[PROVIDER ERROR] {e}\n")
        sys.exit(1)

    # Override voice mode if voice hardware is unavailable
    if health.text_only_mode and args.voice:
        print(
            "[INFO] Microphone/STT unavailable — switching to text mode.\n"
        )
        args.voice = False

    from core.agent import Agent

    agent = Agent(provider=provider)

    if args.voice:
        _run_voice_mode(agent)
    else:
        _run_text_mode(agent)


def _run_text_mode(agent):
    """Text-based chat loop with conversation overlay."""
    from core.reminder_scheduler import ReminderScheduler
    from core.hotkey import HotkeyListener
    from core.timing import TurnTimer

    overlay = ConversationOverlay()
    overlay.start()

    scheduler = None
    if config.REMINDERS_ENABLED:
        scheduler = ReminderScheduler()
        scheduler.start()

    def _hotkey_text_cb():
        print("\n[Hotkey] Type your command: ", end="", flush=True)

    hotkey = HotkeyListener(callback=_hotkey_text_cb)
    hotkey.start()

    print("PAI is ready. Type your message and press Enter.")
    print("Commands: 'reset' to clear history | 'exit' to quit\n")

    try:
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

            tt = TurnTimer()
            with tt.stage("agent"):
                response = agent.chat(user_input)
            tt.log_summary()
            print(f"\nPAI: {response}\n")
            overlay.update(user_input, response)
    finally:
        hotkey.stop()
        if scheduler is not None:
            scheduler.stop()
        overlay.close()


def _run_voice_mode(agent):
    """Voice input loop using wake word + STT, with TTS, overlay, and notifications."""
    from voice.pipeline import VoicePipeline
    from core.exceptions import AudioError
    from core.reminder_scheduler import ReminderScheduler
    from core.hotkey import HotkeyListener
    from core.timing import TurnTimer

    # Load TTS engine — returns None if disabled or on failure
    tts = get_tts()
    if config.TTS_ENABLED and tts is None:
        logger.warning(
            "TTS was enabled but failed to load; continuing without speech output"
        )

    # Start conversation overlay
    overlay = ConversationOverlay()
    overlay.start()

    scheduler = None
    if config.REMINDERS_ENABLED:
        scheduler = ReminderScheduler()
        scheduler.start()

    print(f"Voice mode active. Say '{config.WAKE_WORD}' to speak.")
    print("Press Ctrl+C to quit.\n")

    try:
        pipeline = VoicePipeline()
        pipeline.load()
    except AudioError as e:
        print(f"\n[VOICE ERROR] {e}\n")
        if scheduler is not None:
            scheduler.stop()
        overlay.close()
        sys.exit(1)

    hotkey = HotkeyListener(callback=pipeline.listen_once)
    hotkey.start()

    def handle_transcription(text: str):
        print(f"\nYou: {text}")
        tt = TurnTimer()
        with tt.stage("agent"):
            response = agent.chat(text)
        tt.log_summary()
        print(f"\nPAI: {response}\n")
        overlay.update(text, response)
        notify(response)
        if tts is not None:
            tts.speak(response)

    try:
        pipeline.run_forever(handle_transcription, tts=tts)
    finally:
        hotkey.stop()
        if scheduler is not None:
            scheduler.stop()
        overlay.close()


if __name__ == "__main__":
    main()
