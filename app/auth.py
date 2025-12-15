"""Authentication helpers for the Streamlit app.

This module intentionally uses a fixed salt to keep the demo deterministic. Do
not reuse this approach in production.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from typing import Mapping, MutableMapping, Optional

from .repo import UserRepository

DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "changeit"
_SALT = b"hep-b-static-salt"
_ITERATIONS = 120_000


def get_env_credentials() -> tuple[str, str]:
    """Read credentials from environment or return defaults."""
    username = os.getenv("AUTH_USER", DEFAULT_USERNAME)
    password = os.getenv("AUTH_PASSWORD", DEFAULT_PASSWORD)
    return username, password


def hash_password(password: str) -> str:
    """Derive a pbkdf2-hmac hash for the provided password."""
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        _SALT,
        _ITERATIONS,
    )
    return digest.hex()


def verify_password(password: str, expected_hash: str) -> bool:
    """Constant-time password comparison."""
    candidate = hash_password(password)
    return secrets.compare_digest(candidate, expected_hash)


def bootstrap_user(repo: UserRepository) -> str:
    """Ensure the configured user exists in the repository."""
    username, password = get_env_credentials()
    password_hash = hash_password(password)

    existing = repo.get_user_by_username(username)
    if existing is None:
        repo.create_user(username, password_hash)
    elif not secrets.compare_digest(existing.password_hash, password_hash):
        repo.update_user_password(username, password_hash)

    return username


def init_session_state(session_state: MutableMapping[str, Optional[str]]) -> None:
    """Initialize session container for authentication."""
    session_state.setdefault("user", None)


def is_authenticated(session_state: Mapping[str, Optional[str]]) -> bool:
    return bool(session_state.get("user"))


def login_user(
    username: str,
    password: str,
    session_state: MutableMapping[str, Optional[str]],
    repo: UserRepository,
) -> bool:
    """Validate credentials and persist the user in session state."""
    init_session_state(session_state)

    user = repo.get_user_by_username(username)
    if user is None:
        return False

    if not verify_password(password, user.password_hash):
        return False

    session_state["user"] = username
    return True


def logout(session_state: MutableMapping[str, Optional[str]]) -> None:
    """Clear authentication info."""
    session_state["user"] = None
