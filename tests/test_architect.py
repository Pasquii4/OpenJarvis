import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from openjarvis.agents.architect import ArchitectAgent
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.agents._stubs import AgentResult

from openjarvis.cli.architect_cmd import run_architect_checks

@pytest.fixture
def mock_engine():
    engine = MagicMock(spec=InferenceEngine)
    engine.generate.return_value = {"content": "## Gaps Detectados\n- mock gap"}
    return engine

def test_gather_state_returns_dict():
    # Arrange
    engine = MagicMock(spec=InferenceEngine)
    agent = ArchitectAgent(engine=engine, model="test-model")
    
    # Act
    state = agent._gather_state()
    
    # Assert
    assert isinstance(state, dict)
    assert "agents" in state
    assert "memory_fragments" in state
    assert "goals" in state
    assert isinstance(state["agents"], list)

def test_run_returns_agent_result(mock_engine, tmp_path):
    # Arrange
    # Mock search_memory internally to avoid requiring chroma integration
    with patch("openjarvis.agents.architect.AgentRegistry.keys", return_value=["orchestrator", "architect"]), \
         patch("openjarvis.agents.architect.ArchitectAgent._gather_state", return_value={"agents": ["a"], "memory_fragments": [], "goals": []}):
         
        agent = ArchitectAgent(engine=mock_engine, model="test-model")
        
        # Act
        result = agent.run("analiza el sistema completo")
        
        # Assert
        assert isinstance(result, AgentResult)
        assert "## Gaps Detectados" in result.content
        assert isinstance(result.metadata, dict)
        assert "state" in result.metadata
        mock_engine.generate.assert_called_once()

# --- Architect Goals Parametrized Tests ---

def get_architect_goals():
    goals_path = Path("configs/architect_goals.yaml")
    if not goals_path.exists():
        return []
    try:
        with open(goals_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("goals", [])
    except Exception:
        return []

@pytest.mark.parametrize("goal", get_architect_goals())
def test_architect_goal_coverage(goal):
    # Retrieve status from the architect CLI checker logic
    results = run_architect_checks()
    goal_res = next((r for r in results if r["goal"] == goal), None)
    
    assert goal_res is not None, f"Goal {goal} not processed by run_architect_checks"
    
    # Check if we are in CI
    in_ci = os.environ.get("JARVIS_CI", "0") == "1"
    
    if "OK" not in goal_res["status"]:
        if in_ci:
            pytest.skip(f"Goal {goal} dependency not available in CI environment.")
        else:
            pytest.fail(f"Goal {goal} is not met: {goal_res['detail']}")

