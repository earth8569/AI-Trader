"""Logging bootstrap — concise+colored stderr, full-detail rotating file.

Goals
-----
* CLI must be scannable at a glance: errors red with `✗`, successes green with
  `✓`, warnings yellow with `!`, normal events dim/cyan.
* HOLD spam is filtered out of the console (still in the file).
* Section separators (`───`) bracket each tick so a user can tell where one
  cycle ends and the next begins.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

# --- ANSI color setup --------------------------------------------------------
# Windows 10+ supports ANSI escape codes in cmd/PowerShell, but VT processing
# must be enabled. The empty os.system("") triggers it once per session.
_ANSI_OK = True
if os.name == "nt":
    try:
        os.system("")
    except Exception:
        _ANSI_OK = False
if not sys.stderr.isatty():
    _ANSI_OK = False


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _ANSI_OK else s


RESET   = "\033[0m" if _ANSI_OK else ""
DIM     = "\033[2m" if _ANSI_OK else ""
RED     = "\033[31m" if _ANSI_OK else ""
GREEN   = "\033[32m" if _ANSI_OK else ""
YELLOW  = "\033[33m" if _ANSI_OK else ""
CYAN    = "\033[36m" if _ANSI_OK else ""
GRAY    = "\033[90m" if _ANSI_OK else ""
BOLD    = "\033[1m" if _ANSI_OK else ""

LEVEL_STYLE = {
    logging.DEBUG:    (GRAY,   "."),
    logging.INFO:     (CYAN,   "."),
    logging.WARNING:  (YELLOW, "!"),
    logging.ERROR:    (RED,    "x"),
    logging.CRITICAL: (RED + BOLD, "X"),
}

# message-content cues — when a log line carries one of these tokens, override
# the icon/color so OPENs are obviously different from generic INFO.
CONTENT_CUES = [
    ("OPENED ",    GREEN,  "^"),
    ("CLOSED ",    GREEN,  "v"),
    ("OPEN_BUY",   GREEN,  "^"),
    ("OPEN_SELL",  GREEN,  "v"),
    ("CLOSE_",     CYAN,   "#"),
    ("RECONCILED", CYAN,   "~"),
    ("PROTECTED",  GREEN,  "+"),
    ("TRAIL",      GREEN,  ">"),
    ("OCO placed", GREEN,  "+"),
    ("VETO",       YELLOW, "!"),
    ("FAILED",     RED,    "x"),
]


class ConsoleFormatter(logging.Formatter):
    """Compact: HH:MM:SS  ICON  message"""

    def format(self, record: logging.LogRecord) -> str:
        color, icon = LEVEL_STYLE.get(record.levelno, (RESET, "·"))
        msg = record.getMessage()
        for token, c, i in CONTENT_CUES:
            if token in msg:
                color, icon = c, i
                break
        ts = self.formatTime(record, "%H:%M:%S")
        if record.levelno >= logging.ERROR and record.exc_info:
            msg = msg + "\n" + self.formatException(record.exc_info)
        return f"{GRAY}{ts}{RESET} {color}{icon}{RESET} {msg}"


class HoldFilter(logging.Filter):
    """Drop noisy per-symbol HOLD lines from the console (kept in file)."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        msg = record.getMessage()
        if "status=HOLD" in msg or " HOLD " in msg or msg.endswith(" HOLD"):
            return False
        # also suppress the verbose tick equity line — replaced by the
        # banner emitted from the trader itself
        if "tick equity=" in msg:
            return False
        return True


def setup_logging(log_path: str | Path = "logs/trader.log", level: int = logging.INFO) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)   # let handlers decide what to show

    # console: concise, colored, INFO+ filtered by HoldFilter
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(ConsoleFormatter())
    console.addFilter(HoldFilter())
    root.addHandler(console)

    # file: keep the full original format + DEBUG-level detail for diagnostics
    file_h = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_h)

    # quiet noisy third-party libs on the console
    for noisy in ("urllib3", "requests", "requests.packages.urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def banner(text: str, color: str = CYAN, char: str = "-", width: int = 60) -> str:
    """Return a thin separator line for tick start/end. Use via print()."""
    pad = max(0, width - len(text) - 2)
    return f"{color}{char * 3} {text} {char * pad}{RESET}" if _ANSI_OK else f"--- {text} {'-' * pad}"
