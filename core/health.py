"""
core/health.py

PAI startup component health checker.

run_health_check() probes each major component and returns a HealthReport.
Components are classified OK, DEGRADED, or MISSING.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

from core.logger import logger


class ComponentStatus(str, Enum):
    """Health classification for a single PAI component.

    Values:
        OK: Component is present and fully functional.
        DEGRADED: Component works in a reduced or fallback mode.
        MISSING: Component is unavailable.
    """

    OK       = "ok"
    DEGRADED = "degraded"
    MISSING  = "missing"


@dataclass
class ComponentResult:
    """Outcome of probing one component during the health check.

    Attributes:
        name: Stable identifier for the component (e.g. "llm_provider").
        status: The component's :class:`ComponentStatus`.
        detail: Optional human-readable note (path, version, error message).
    """

    name:   str
    status: ComponentStatus
    detail: str = ""

    def __str__(self) -> str:
        """Render the result as an icon-prefixed single line for the summary."""
        icon   = {"ok": "✓", "degraded": "⚠", "missing": "✗"}[self.status]
        suffix = f" — {self.detail}" if self.detail else ""
        return f"  {icon} {self.name}{suffix}"


@dataclass
class HealthReport:
    """Aggregated results of a startup health check.

    Collects per-component results and exposes derived properties that the
    CLI uses to decide whether to abort, drop to text-only mode, or disable
    individual features.

    Attributes:
        components: List of :class:`ComponentResult` entries, one per probe.
    """

    components: list[ComponentResult] = field(default_factory=list)

    @property
    def critical_failure(self) -> bool:
        """Return True if a component required to start at all is missing.

        Returns:
            True when the LLM provider or memory database is MISSING.
        """
        _CRITICAL = {"llm_provider", "memory_db"}
        return any(
            c.status == ComponentStatus.MISSING and c.name in _CRITICAL
            for c in self.components
        )

    @property
    def text_only_mode(self) -> bool:
        """Return True if voice dependencies are unavailable.

        Returns:
            True when the microphone or STT model is MISSING, meaning voice
            mode cannot run and PAI should fall back to text input.
        """
        _VOICE_DEPS = {"microphone", "stt_model"}
        return any(
            c.status == ComponentStatus.MISSING and c.name in _VOICE_DEPS
            for c in self.components
        )

    @property
    def tts_disabled(self) -> bool:
        """Return True if the text-to-speech engine is missing.

        Returns:
            True when the ``tts_engine`` component is MISSING.
        """
        return any(
            c.name == "tts_engine" and c.status == ComponentStatus.MISSING
            for c in self.components
        )

    @property
    def browser_disabled(self) -> bool:
        """Return True if the browser component is not fully functional.

        Returns:
            True when the ``browser`` component has any status other than OK.
        """
        return any(
            c.name == "browser" and c.status != ComponentStatus.OK
            for c in self.components
        )

    def summary(self) -> str:
        """Build a multi-line, human-readable summary of all components.

        Returns:
            A formatted report string with one line per component plus a
            trailing FATAL or INFO note when a critical or voice dependency
            is missing.
        """
        lines = ["PAI Component Health Check:"]
        lines.extend(str(c) for c in self.components)
        if self.critical_failure:
            lines.append("\n  [FATAL] Critical component missing — cannot start.")
        elif self.text_only_mode:
            lines.append(
                "\n  [INFO] Running in text-only mode "
                "(microphone or STT unavailable)."
            )
        return "\n".join(lines)


def run_health_check() -> HealthReport:
    """Probe all major PAI components and return an aggregated report.

    Runs each component check in turn (provider, memory DB, microphone, STT,
    TTS, browser) and logs each result at DEBUG level. When
    ``config.SKIP_HEALTH_CHECK`` is set, returns an empty report immediately.

    Returns:
        A :class:`HealthReport` populated with one result per component, or an
        empty report if the check was skipped.
    """
    import config

    report = HealthReport()

    if config.SKIP_HEALTH_CHECK:
        logger.info("Health check skipped (SKIP_HEALTH_CHECK=true)")
        return report

    _check_llm_provider(report)
    _check_memory_db(report)
    _check_microphone(report)
    _check_stt(report)
    _check_tts(report)
    _check_browser(report)

    for c in report.components:
        logger.debug(f"Health: {c}")

    return report


def _check_llm_provider(report: HealthReport) -> None:
    """Validate provider configuration and append the result to the report.

    Args:
        report: The report to append the ``llm_provider`` result to.

    Side effects:
        Appends a :class:`ComponentResult`; OK when configuration validates,
        MISSING when :func:`config.validate_config` raises ``ConfigError``.
    """
    import config
    from core.exceptions import ConfigError
    try:
        config.validate_config()
        report.components.append(
            ComponentResult("llm_provider", ComponentStatus.OK, config.LLM_PROVIDER)
        )
    except ConfigError as e:
        report.components.append(
            ComponentResult("llm_provider", ComponentStatus.MISSING, str(e))
        )


def _check_memory_db(report: HealthReport) -> None:
    """Verify the SQLite memory database can be created and opened.

    Expands ``config.MEMORY_DB_PATH``, ensures the parent directory exists,
    and opens then closes a connection to confirm writability.

    Args:
        report: The report to append the ``memory_db`` result to.

    Side effects:
        May create the database's parent directory. Appends a
        :class:`ComponentResult` (OK or MISSING with the error detail).
    """
    import sqlite3
    from pathlib import Path
    import config

    db_path = Path(config.MEMORY_DB_PATH).expanduser()
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.close()
        report.components.append(
            ComponentResult("memory_db", ComponentStatus.OK, str(db_path))
        )
    except Exception as e:
        report.components.append(
            ComponentResult("memory_db", ComponentStatus.MISSING, str(e))
        )


def _check_microphone(report: HealthReport) -> None:
    """Detect available audio input devices via sounddevice.

    Args:
        report: The report to append the ``microphone`` result to.

    Side effects:
        Appends a :class:`ComponentResult`: OK with the input-device count,
        or MISSING when no inputs are found or ``sounddevice`` is unavailable.
    """
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        if inputs:
            report.components.append(ComponentResult(
                "microphone", ComponentStatus.OK,
                f"{len(inputs)} input device(s) found",
            ))
        else:
            report.components.append(ComponentResult(
                "microphone", ComponentStatus.MISSING, "no input devices detected"
            ))
    except Exception as e:
        report.components.append(
            ComponentResult("microphone", ComponentStatus.MISSING, str(e))
        )


def _check_stt(report: HealthReport) -> None:
    """Check whether the faster-whisper speech-to-text backend is installed.

    Args:
        report: The report to append the ``stt_model`` result to.

    Side effects:
        Appends a :class:`ComponentResult`: OK with the configured model size,
        or MISSING when ``faster_whisper`` cannot be imported.
    """
    import config
    try:
        import faster_whisper  # noqa: F401
        report.components.append(ComponentResult(
            "stt_model", ComponentStatus.OK,
            f"faster-whisper ({config.STT_SIZE})",
        ))
    except ImportError:
        report.components.append(ComponentResult(
            "stt_model", ComponentStatus.MISSING, "faster-whisper not installed"
        ))


def _check_tts(report: HealthReport) -> None:
    """Check the configured text-to-speech engine and any fallback.

    For the ``kokoro`` engine, probes kokoro first and reports DEGRADED if
    only the ``pyttsx3`` fallback is available. For ``pyttsx3``, probes it
    directly. Any other engine name is reported as DEGRADED.

    Args:
        report: The report to append the ``tts_engine`` result to.

    Side effects:
        Appends a :class:`ComponentResult` describing engine availability.
    """
    import config
    engine = config.TTS_ENGINE

    if engine == "kokoro":
        try:
            import kokoro  # noqa: F401
            report.components.append(
                ComponentResult("tts_engine", ComponentStatus.OK, "kokoro")
            )
        except ImportError:
            try:
                import pyttsx3  # noqa: F401
                report.components.append(ComponentResult(
                    "tts_engine", ComponentStatus.DEGRADED,
                    "kokoro missing — pyttsx3 fallback active",
                ))
            except ImportError:
                report.components.append(ComponentResult(
                    "tts_engine", ComponentStatus.MISSING,
                    "kokoro and pyttsx3 both unavailable",
                ))
    elif engine == "pyttsx3":
        try:
            import pyttsx3  # noqa: F401
            report.components.append(
                ComponentResult("tts_engine", ComponentStatus.OK, "pyttsx3")
            )
        except ImportError:
            report.components.append(ComponentResult(
                "tts_engine", ComponentStatus.MISSING, "pyttsx3 not installed"
            ))
    else:
        report.components.append(ComponentResult(
            "tts_engine", ComponentStatus.DEGRADED, f"unknown engine: {engine}"
        ))


def _check_browser(report: HealthReport) -> None:
    """Locate a usable browser binary for the configured browser app.

    Honors an explicit ``config.BROWSER_BINARY`` path when it points to a
    file; otherwise searches PATH for known binary names matching
    ``config.BROWSER_APP`` (chrome, firefox, edge, or the raw name).

    Args:
        report: The report to append the ``browser`` result to.

    Side effects:
        Appends a :class:`ComponentResult`: OK with the resolved path, or
        DEGRADED when no binary is found (disabling the browser tool).
    """
    import os
    import shutil
    import config

    if config.BROWSER_BINARY and os.path.isfile(config.BROWSER_BINARY):
        report.components.append(ComponentResult(
            "browser", ComponentStatus.OK, config.BROWSER_BINARY
        ))
        return

    browser_name = config.BROWSER_APP.lower()
    binary_candidates = {
        "chrome":   ["google-chrome", "chrome", "chromium", "chromium-browser"],
        "firefox":  ["firefox"],
        "edge":     ["microsoft-edge", "msedge"],
    }
    candidates = binary_candidates.get(browser_name, [browser_name])

    found = next((shutil.which(c) for c in candidates if shutil.which(c)), None)

    if found:
        report.components.append(
            ComponentResult("browser", ComponentStatus.OK, found)
        )
    else:
        report.components.append(ComponentResult(
            "browser", ComponentStatus.DEGRADED,
            f"{browser_name} not found on PATH — browser tool disabled",
        ))
