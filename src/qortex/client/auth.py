"""API token resolution for OpenNeuro.

Priority order (highest → lowest):
  1. Explicit token passed to ``OpenNeuroClient(token=...)``
  2. Environment variable ``QORTEX_API_TOKEN``
  3. Config file  ``~/.config/qortex/credentials.json``
  4. Anonymous (no token)
"""

from __future__ import annotations

import getpass
import json
import os
import stat
from pathlib import Path

import platformdirs

from qortex.core.exceptions import AuthError

_CRED_DIR = Path(
    platformdirs.user_config_dir("qortex", appauthor=False, roaming=True)
)
_CRED_FILE = _CRED_DIR / "credentials.json"


# ── Public API ────────────────────────────────────────────────────────────────

def resolve_token(explicit: str | None = None) -> str | None:
    """Return the best available API token, or None for anonymous access."""
    if explicit:
        return explicit

    # Env var (also read by QortexConfig, but we check directly here so the
    # client can be used without a full config object)
    env = os.environ.get("QORTEX_API_TOKEN") or os.environ.get("OPENNEURO_API_TOKEN")
    if env:
        return env

    return _load_from_file()


def save_token(token: str) -> Path:
    """Persist an API token to the credentials file (chmod 600)."""
    _CRED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"apikey": token}
    _CRED_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _CRED_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return _CRED_FILE


def prompt_and_save() -> str:
    """Interactively prompt the user for an API token and persist it."""
    print(
        "Please log in to OpenNeuro (https://openneuro.org) and obtain an API key:\n"
        "  My Account → API Key\n"
    )
    token = getpass.getpass("OpenNeuro API key (input hidden): ").strip()
    if not token:
        raise AuthError("No API key entered.")
    save_token(token)
    print(f"Token saved to {_CRED_FILE}")
    return token


def delete_token() -> None:
    """Remove the saved credentials file."""
    if _CRED_FILE.exists():
        _CRED_FILE.unlink()


def has_token() -> bool:
    """Return True if any token is available (does not validate it)."""
    return resolve_token() is not None


# ── Internal ──────────────────────────────────────────────────────────────────

def _load_from_file() -> str | None:
    if not _CRED_FILE.exists():
        return None
    try:
        data = json.loads(_CRED_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("apikey") or data.get("token") or None
