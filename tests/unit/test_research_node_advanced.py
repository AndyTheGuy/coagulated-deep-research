import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.models import ResearcherInput, SubQuestion, SearchResult, VerifiedSource
from core.nodes.research import researcher_node

@pytest.fixture
def sample_researcher_input():
    sub_q = SubQuestion(
        id="q_advanced",
        question="What is the latest achievement of surface codes?",
        status="pending"
    )
    return ResearcherInput(
        sub_question=sub_q,
        topic="Quantum Computing",
        constraints=["focus on fault-tolerance"]
    )

@pytest.mark.asyncio
async def test_advanced_researcher_node_full_pipeline_success(sample_researcher_input):
    """Test advanced researcher node full integration path under standard success conditions."""
    
    # Mock search results returned by search_searxng
    mock_search_results = [
        SearchResult(title="Surface Code Milestones", url="https://quantum-example.com/milestones", content="Milestone content."),
        SearchResult(title="Fault Tolerant Surface Codes", url="https://quantum-example.com/ft", content="FT content.")
    ]
    
    # Mock embeddings
    mock_embeddings_instance = MagicMock()
    mock_embeddings_instance.aembed_documents = AsyncMock(return_value=[
        [1.0] + [0.0]*767,
        [0.0] + [1.0] + [0.0]*766
    ])
    
    # Mock PCTSEngine, MangoRouter, and BrowserExplorer
    mock_mcts_instance = MagicMock()
    mock_mcts_instance.search = AsyncMock(return_value=(
        "Analyze surface code milestones", 
        ["surface codes latest achievements", "fault tolerance thresholds"]
    ))
    
    mock_explore_res_1 = {
        "success": True, 
        "title": "Surface Code Milestones", 
        "content": "Excellent factual density about surface code thresholds reaching 1%!", 
        "method": "puppeteer"
    }
    mock_explore_res_2 = {
        "success": True, 
        "title": "Fault Tolerant Surface Codes", 
        "content": "High fidelity qubits were coupled.", 
        "method": "scraper_fallback"
    }
    
    mock_explorer_instance = MagicMock()
    mock_explorer_instance.explore_url = AsyncMock(side_effect=[mock_explore_res_1, mock_explore_res_2])
    
    # Mock LLMRouter for synthesis
    mock_response = MagicMock()
    mock_response.content = "Latest research shows surface code physical thresholds have reached 1% with active error correction."
    mock_response.usage_metadata = {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
    
    mock_router_instance = MagicMock()
    mock_router_instance.ainvoke = AsyncMock(return_value=mock_response)
    mock_router_instance.token_usage = {
        "freellmapi": {"calls": 1, "input_tokens": 50, "output_tokens": 20}
    }
    
    mock_mango_instance = MagicMock()
    mock_mango_instance.select_url.side_effect = [
        "https://quantum-example.com/milestones",
        "https://quantum-example.com/ft",
        None
    ]
    
    # Setup patches
    with patch("core.nodes.research.search_searxng", new_callable=AsyncMock) as mock_searxng, \
         patch("core.nodes.research.get_embeddings", return_value=mock_embeddings_instance), \
         patch("core.nodes.research.PCTSEngine", return_value=mock_mcts_instance), \
         patch("core.nodes.research.BrowserExplorer", return_value=mock_explorer_instance), \
         patch("core.nodes.research.evaluate_scraped_relevance", new_callable=AsyncMock) as mock_eval, \
         patch("core.nodes.research.LLMRouter", return_value=mock_router_instance), \
         patch("core.nodes.research.MangoRouter", return_value=mock_mango_instance), \
         patch("core.nodes.research.MCPHub") as mock_hub_class:
         
        mock_searxng.return_value = mock_search_results
        mock_eval.side_effect = [0.95, 0.75] # Factual density rewards
        
        mock_hub = MagicMock()
        mock_hub.connect_all = AsyncMock()
        mock_hub.shutdown = AsyncMock()
        mock_hub_class.return_value = mock_hub
        
        # Execute node
        result = await researcher_node(sample_researcher_input)
        
        # 1. Verify schema
        assert "sub_questions_state" in result
        assert "search_results" in result
        assert "verified_sources" in result
        assert "token_usage" in result
        assert "logs" in result
        assert "errors" in result
        
        # 2. Verify sub-question updates
        assert len(result["sub_questions_state"]) == 1
        updated_sub_q = result["sub_questions_state"][0]
        assert updated_sub_q.id == "q_advanced"
        assert updated_sub_q.status == "completed"
        assert "physical thresholds have reached 1%" in updated_sub_q.results_summary
        
        # 3. Verify MCTS Planning was utilized
        assert any("MCTS generation" in log or "MCTS Selected Intent" in log or "MCTS generated" in log for log in result["logs"])
        mock_mcts_instance.search.assert_called_once_with(
            sub_question="What is the latest achievement of surface codes?",
            topic="Quantum Computing",
            max_iterations=2
        )
        
        # 4. Verify parallel search executed on MCTS queries
        assert mock_searxng.call_count == 2 # Called for both MCTS queries
        
        # 5. Verify Mango selected and explorer scraped the correct URLs
        assert len(result["verified_sources"]) == 2
        assert result["verified_sources"][0].url == "https://quantum-example.com/milestones"
        assert result["verified_sources"][0].accessible is True
        assert result["verified_sources"][0].content == "Excellent factual density about surface code thresholds reaching 1%!"
        
        assert result["verified_sources"][1].url == "https://quantum-example.com/ft"
        assert result["verified_sources"][1].accessible is True
        assert result["verified_sources"][1].content == "High fidelity qubits were coupled."
        
        # 6. Verify logging statements
        assert any("Completed starting-point URL selection via Mango" in log for log in result["logs"])
        assert any("Scraped URL:" in log and "Mango Reward:" in log for log in result["logs"])
