"""Small progress-printing helper for long example workflows."""

from __future__ import annotations

from datetime import datetime


def progress(message: str) -> None:
    """Print a timestamped progress message and flush immediately."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
