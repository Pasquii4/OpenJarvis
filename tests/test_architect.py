import pytest
from unittest.mock import MagicMock, patch
from openjarvis.agents.architect import ArchitectAgent
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.agents._stubs import AgentResult

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
