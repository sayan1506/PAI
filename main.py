"""
PAI — Personal Assistance and Intelligence

CLI entry point that initializes configuration, creates the Agent,
and runs the interactive chat loop.
"""

import sys

from core.logger import logger
from core.exceptions import ConfigError, ProviderError
import config


def main():
    """Main entry point for the PAI assistant."""
    print(f"\n{'='*50}")
    print(f"  PAI — Personal Assistance and Intelligence")
    print(f"  Provider: {config.LLM_PROVIDER.upper()}")
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


if __name__ == "__main__":
    main()
