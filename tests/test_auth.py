from pathlib import Path
from typing import List

import pytest

from app import auth
from app.db import init_db
from app.repo import UserRepository
from app.main import render_login, require_auth


class StubStreamlit:
    """Minimal stub for Streamlit interactions used in authentication."""

    def __init__(self, inputs: List[str], button_presses: List[bool]) -> None:
        self.session_state = {}
        self._inputs = inputs
        self._button_presses = button_presses
        self.messages = []
        self.stopped = False

    def subheader(self, message: str) -> None:
        self.messages.append(("subheader", message))

    def text_input(self, label: str, type: str | None = None) -> str:
        return self._inputs.pop(0)

    def button(self, label: str) -> bool:
        return self._button_presses.pop(0)

    def success(self, message: str) -> None:
        self.messages.append(("success", message))

    def error(self, message: str) -> None:
        self.messages.append(("error", message))

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture()
def schema_path() -> Path:
    return Path(__file__).resolve().parent.parent / "app" / "schema.sql"


@pytest.fixture()
def user_repo(tmp_path, schema_path) -> UserRepository:
    db_path = tmp_path / "auth.db"
    db = init_db(db_path, schema_path)
    return UserRepository(db)


def test_env_defaults(monkeypatch):
    monkeypatch.delenv("AUTH_USER", raising=False)
    monkeypatch.delenv("AUTH_PASSWORD", raising=False)

    username, password = auth.get_env_credentials()
    assert username == auth.DEFAULT_USERNAME
    assert password == auth.DEFAULT_PASSWORD


def test_hash_and_verify_roundtrip():
    password = "s3cret!"
    hashed = auth.hash_password(password)
    assert hashed != password
    assert auth.verify_password(password, hashed) is True
    assert auth.verify_password("wrong", hashed) is False


def test_bootstrap_creates_and_updates_user(monkeypatch, user_repo):
    monkeypatch.setenv("AUTH_USER", "alice")
    monkeypatch.setenv("AUTH_PASSWORD", "pw1")

    created_username = auth.bootstrap_user(user_repo)
    assert created_username == "alice"
    created = user_repo.get_user_by_username("alice")
    assert created is not None
    assert created.password_hash == auth.hash_password("pw1")

    # Changing the password should update the stored hash.
    monkeypatch.setenv("AUTH_PASSWORD", "pw2")
    auth.bootstrap_user(user_repo)
    updated = user_repo.get_user_by_username("alice")
    assert updated is not None
    assert updated.password_hash == auth.hash_password("pw2")


def test_login_and_logout_flow(monkeypatch, user_repo):
    monkeypatch.setenv("AUTH_USER", "bob")
    monkeypatch.setenv("AUTH_PASSWORD", "pw3")
    auth.bootstrap_user(user_repo)

    session_state: dict[str, str | None] = {}
    assert auth.login_user("bob", "pw3", session_state, user_repo) is True
    assert session_state["user"] == "bob"

    auth.logout(session_state)
    assert session_state["user"] is None

    assert auth.login_user("bob", "wrong", session_state, user_repo) is False
    assert session_state["user"] is None


def test_login_rejects_unknown_user(user_repo):
    session_state: dict[str, str | None] = {}
    assert auth.login_user("ghost", "pw", session_state, user_repo) is False
    assert session_state["user"] is None


def test_hash_password_requires_string():
    with pytest.raises(TypeError):
        auth.hash_password(None)  # type: ignore[arg-type]


def test_render_login_and_guard(monkeypatch, user_repo):
    monkeypatch.setenv("AUTH_USER", "cathy")
    monkeypatch.setenv("AUTH_PASSWORD", "pw4")
    auth.bootstrap_user(user_repo)

    # First attempt: wrong password should not authenticate.
    stub = StubStreamlit(inputs=["cathy", "badpw"], button_presses=[True])
    assert render_login(user_repo, st_module=stub) is False
    assert auth.is_authenticated(stub.session_state) is False

    # Second attempt: correct password and guard should pass, stop should not trigger.
    stub = StubStreamlit(inputs=["cathy", "pw4"], button_presses=[True])
    assert render_login(user_repo, st_module=stub) is True

    guard_stub = StubStreamlit(inputs=["cathy", "pw4"], button_presses=[True])
    require_auth(user_repo, st_module=guard_stub)
    assert guard_stub.stopped is False
