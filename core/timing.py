"""
core/timing.py

Per-stage latency instrumentation for PAI.

Provides two utilities:

  Timer        — context manager that measures one stage and logs elapsed
                 time at DEBUG level immediately on exit.

  TurnTimer    — accumulates timings for all stages of a single conversation
                 turn, then emits a single INFO-level summary at the end.
                 Call log_summary() after the turn is fully complete.

Typical turn stages: stt, memory_fetch, llm, tool_1 ... tool_N, tts_start

Usage:
    from core.timing import Timer, TurnTimer

    # One-off stage:
    with Timer("stt"):
        text = stt.transcribe(audio)

    # Full turn:
    tt = TurnTimer()
    with tt.stage("stt"):
        text = stt.transcribe(audio)
    with tt.stage("llm"):
        response = agent.chat(text)
    tt.log_summary()
    # → [TURN] stt=381ms  llm=719ms  | total=1100ms
"""

import time
from contextlib import contextmanager

from core.logger import logger


@contextmanager
def Timer(stage_name: str):
    """
    Context manager that logs elapsed time for a single named stage.

    Logs at DEBUG level on exit:
        [TIMING] <stage_name>: <elapsed>ms

    Args:
        stage_name: Human-readable label for the stage being timed.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"[TIMING] {stage_name}: {elapsed_ms:.1f}ms")


class TurnTimer:
    """
    Accumulates per-stage timings for one conversation turn.

    Usage:
        tt = TurnTimer()
        with tt.stage("stt"):   text     = transcribe(audio)
        with tt.stage("llm"):   response = agent.chat(text)
        tt.log_summary()

    Logged summary format (INFO level):
        [TURN] stt=381ms  llm=719ms  | total=1100ms

    A DEBUG warning is also emitted if the total exceeds the 2500ms
    latency budget from the project spec (Appendix B).
    """

    def __init__(self) -> None:
        self._stages: list[tuple[str, float]] = []
        self._turn_start = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        """
        Context manager for one stage within this turn.

        Appends (name, elapsed_ms) to the internal list on exit.
        No output until log_summary() is called.

        Args:
            name: Stage label (e.g. "stt", "llm", "tool_1").
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._stages.append((name, elapsed_ms))

    def log_summary(self) -> None:
        """
        Emit one INFO-level line summarising all recorded stages.

        Format: [TURN] stt=381ms  llm=719ms  tool_1=42ms  | total=1142ms

        Also logs a DEBUG warning if total > 2500ms (the spec's latency
        budget target for simple queries).
        """
        total_ms = (time.perf_counter() - self._turn_start) * 1000
        stage_str = "  ".join(
            f"{name}={ms:.0f}ms" for name, ms in self._stages
        )
        logger.info(f"[TURN] {stage_str}  | total={total_ms:.0f}ms")

        if total_ms > 2500 and self._stages:
            slowest = max(self._stages, key=lambda x: x[1])
            logger.debug(
                f"[TIMING] Turn exceeded 2500ms budget — slowest stage: "
                f"{slowest[0]}={slowest[1]:.0f}ms"
            )

    def stages(self) -> list[tuple[str, float]]:
        """Return a copy of all recorded (name, elapsed_ms) tuples."""
        return list(self._stages)
