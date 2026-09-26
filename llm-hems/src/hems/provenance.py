"""Run provenance: which code produced a results folder.

`provenance()` returns the git commit, whether the working tree was dirty,
the branch, a UTC timestamp, and the command line; `write_provenance(out)`
writes it as `provenance.json` into a results folder. Every results-writing
experiment script calls it, so a number in `results/` can always be traced
back to the exact code (and whether uncommitted edits were in play).
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.rstrip("\n")  # not strip(): porcelain lines start with a space


def provenance() -> dict:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": (_git("rev-parse", "HEAD") or "").strip() or None,
        "dirty": None if status is None else bool(status),
        "dirty_files": None if not status else [line[3:] for line in status.splitlines()],
        "branch": (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "python": platform.python_version(),
    }


def write_provenance(out_dir: str | Path, extra: dict | None = None) -> dict:
    """Write `provenance.json` (plus any `extra` keys) into `out_dir`; returns it."""
    info = {**provenance(), **(extra or {})}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(info, indent=2))
    return info
