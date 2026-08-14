from collections.abc import Generator
from pathlib import Path

import pytest

from pipeline.core.config import settings as settings_module


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    settings_module.get_settings.cache_clear()
    yield
    settings_module.get_settings.cache_clear()


def configure_project_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", root)
    monkeypatch.setattr(settings_module, "DEFAULT_ENV_FILE", root / ".env")


def test_default_env_file_is_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_project_root(monkeypatch, tmp_path)
    monkeypatch.delenv("SAG_ENV_FILE", raising=False)

    assert settings_module.resolve_env_file() == tmp_path / ".env"


def test_absolute_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_project_root(monkeypatch, tmp_path)
    env_file = tmp_path / "secure.env"
    env_file.write_text("LLM_MODEL=from-absolute-file\n", encoding="utf-8")
    monkeypatch.setenv("SAG_ENV_FILE", str(env_file))

    assert settings_module.resolve_env_file() == env_file.resolve()


def test_relative_override_is_relative_to_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_project_root(monkeypatch, tmp_path)
    env_file = tmp_path / "config" / "test.env"
    env_file.parent.mkdir()
    env_file.write_text("LLM_MODEL=from-relative-file\n", encoding="utf-8")
    monkeypatch.setenv("SAG_ENV_FILE", "config/test.env")

    assert settings_module.resolve_env_file() == env_file.resolve()


def test_missing_explicit_override_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_project_root(monkeypatch, tmp_path)
    monkeypatch.setenv("SAG_ENV_FILE", "missing.env")

    with pytest.raises(FileNotFoundError, match="SAG_ENV_FILE"):
        settings_module.resolve_env_file()


def test_get_settings_reads_selected_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_project_root(monkeypatch, tmp_path)
    env_file = tmp_path / "selected.env"
    env_file.write_text("LLM_MODEL=from-selected-file\n", encoding="utf-8")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("SAG_ENV_FILE", str(env_file))

    assert settings_module.get_settings().llm_model == "from-selected-file"


def test_explicit_env_file_none_still_bypasses_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_project_root(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("LLM_MODEL=must-not-load\n", encoding="utf-8")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    settings = settings_module.Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.llm_model != "must-not-load"
