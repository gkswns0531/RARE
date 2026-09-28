import sys
from pathlib import Path

import pytest

# Add RARE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rare_core.rare_orchestration_service import RareOrchestrator


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM retries back off with time.sleep; tests do not need to wait."""
    monkeypatch.setattr("time.sleep", lambda seconds: None)


@pytest.fixture
def orchestrator(tmp_path: Path) -> RareOrchestrator:
    """Orchestrator whose LLM calls must be replaced by each test."""
    return RareOrchestrator(api_key="test-key", output_dir=str(tmp_path / "outputs"))
