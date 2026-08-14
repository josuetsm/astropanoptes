import pytest


@pytest.fixture(autouse=True)
def _isolate_operational_goto_logs(monkeypatch, tmp_path):
    """Never let a test append synthetic samples to observing-session logs."""
    monkeypatch.setenv(
        "ASTROPANOPTES_GOTO_LOG_DIR",
        str(tmp_path / "goto_logs"),
    )
