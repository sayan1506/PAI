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
    OK       = "ok"
    DEGRADED = "degraded"
    MISSING  = "missing"


@dataclass
class ComponentResult:
    name:   str
    status: ComponentStatus
    detail: str = ""

    def __str__(self) -> str:
        icon   = {"ok": "✓", "degraded": "⚠", "missing": "✗"}[self.status]
        suffix = f" — {self.detail}" if self.detail else ""
        return f"  {icon} {self.name}{suffix}"


@dataclass
class HealthReport:
    components: list[ComponentResult] = field(default_factory=list)

    @property
    def critical_failure(self) -> bool:
        _CRITICAL = {"llm_provider", "memory_db"}
        return any(
            c.status == ComponentStatus.MISSING and c.name in _CRITICAL
            for c in self.components
        )

    @property
    def text_only_mode(self) -> bool:
        _VOICE_DEPS = {"microphone", "stt_model"}
        return any(
            c.status == ComponentStatus.MISSING and c.name in _VOICE_DEPS
            for c in self.components
        )

    @property
    def tts_disabled(self) -> bool:
        return any(
            c.name == "tts_engine" and c.status == ComponentStatus.MISSING
            for c in self.components
        )

    @property
    def browser_disabled(self) -> bool:
        return any(
            c.name == "browser" and c.status != ComponentStatus.OK
            for c in self.components
        )

    def summary(self) -> str:
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
