import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core.models import ResearcherInput, SubQuestion, SearchResult, VerifiedSource
from core.nodes.research import researcher_node

# Create a sample input state
@pytest.fixture
def sample_researcher_input():
    sub_q = SubQuestion(
        id="q1",
        question="What is quantum error correction?",
        status="pending"
    )
    return ResearcherInput(
        sub_question=sub_q,
        topic="Quantum Computing",
        constraints=["focus on surface codes"]
    )

@pytest.mark.asyncio
async def test_researcher_node_success(sample_researcher_input, mock_llm_calls):
    """Test successful researcher node execution with mock search and scrape results."""
    # Unpack the mock LLM calls fixture (mock_vertex, mock_openai)
    mock_vertex, mock_openai = mock_llm_calls
    
    # We will override the mock response for OpenAI (standard tier) specifically to return a synthesis summary
    mock_openai.ainvoke.reset_mock()
    
    # First invocation is query diversification (BULK tier) -> returns variant list
    # Second invocation is summary synthesis (STANDARD tier) -> returns final summary
    mock_response_1 = MagicMock()
    mock_response_1.content = '{"variants": ["quantum error correction codes", "surface codes stability", "FTQC physical qubits"]}'
    mock_response_1.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    
    mock_response_2 = MagicMock()
    mock_response_2.content = "Quantum error correction (QEC) is essential for fault-tolerant quantum computing. Surface codes are currently the leading approach."
    mock_response_2.usage_metadata = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
    
    mock_openai.ainvoke.side_effect = [mock_response_2]

    # Mock search results returned by search_searxng
    mock_search_results = [
        SearchResult(title="Surface Codes Guide", url="https://quantum-example.com/surface", content="Surface code tutorial content."),
        SearchResult(title="QEC Basics", url="https://quantum-example.com/qec", content="Intro to QEC theory.")
    ]
    
    # Mock embeddings to avoid actual sentence-transformers loading in standard test (for speed)
    mock_embeddings_instance = MagicMock()
    mock_embeddings_instance.aembed_documents = AsyncMock(return_value=[
        [1.0] + [0.0]*767,
        [0.0] + [1.0] + [0.0]*766
    ])
    
    # Mocking external calls
    with patch("core.nodes.research.search_searxng", new_callable=AsyncMock) as mock_searxng, \
         patch("core.nodes.research.BrowserExplorer.explore_url", new_callable=AsyncMock) as mock_explore, \
         patch("core.nodes.research.get_embeddings", return_value=mock_embeddings_instance), \
         patch("core.nodes.research.PCTSEngine.search", new_callable=AsyncMock) as mock_mcts, \
         patch("core.nodes.research.evaluate_scraped_relevance", new_callable=AsyncMock) as mock_eval, \
         patch("core.nodes.research.MangoRouter.select_url") as mock_select, \
         patch("core.nodes.research.MCPHub") as mock_hub_class:
         
        mock_searxng.return_value = mock_search_results
        mock_mcts.return_value = ("mock intent", ["quantum error correction codes", "surface codes stability"])
        mock_eval.return_value = 0.8
        mock_select.side_effect = ["https://quantum-example.com/surface", "https://quantum-example.com/qec", None]
        
        mock_hub = MagicMock()
        mock_hub.connect_all = AsyncMock()
        mock_hub.shutdown = AsyncMock()
        mock_hub_class.return_value = mock_hub
        
        # Setup mock explorer responses
        mock_explore.side_effect = [
            {"success": True, "title": "Surface Codes Guide", "content": "Detailed guide on surface codes and physical qubits.", "method": "puppeteer"},
            {"success": True, "title": "QEC Basics", "content": "Thorough introduction to general quantum error correction.", "method": "scraper_fallback"}
        ]
        
        # Execute node
        result = await researcher_node(sample_researcher_input)
        
        # Verify result dictionary schema and contents
        assert "sub_questions_state" in result
        assert "search_results" in result
        assert "verified_sources" in result
        assert "token_usage" in result
        assert "logs" in result
        assert "errors" in result
        
        # SubQuestion completed verification
        assert len(result["sub_questions_state"]) == 1
        updated_sub_q = result["sub_questions_state"][0]
        assert updated_sub_q.id == "q1"
        assert updated_sub_q.status == "completed"
        assert updated_sub_q.assigned_researcher == "researcher_node"
        assert "Surface codes are currently" in updated_sub_q.results_summary
        
        # Search results verification (fused)
        assert len(result["search_results"]) > 0
        
        # Scraped sources verification
        assert len(result["verified_sources"]) == 2
        assert result["verified_sources"][0].accessible is True
        assert result["verified_sources"][0].url == "https://quantum-example.com/surface"
        assert result["verified_sources"][0].content == "Detailed guide on surface codes and physical qubits."
        
        # Token usage verification
        assert result["token_usage"]["freellmapi"]["calls"] == 1
        assert result["token_usage"]["freellmapi"]["input_tokens"] == 20
        assert result["token_usage"]["freellmapi"]["output_tokens"] == 10
        
        # Verify logger messages
        assert any("Successfully researched and synthesized findings" in log for log in result["logs"])

@pytest.mark.asyncio
async def test_researcher_node_no_accessible_sources(sample_researcher_input, mock_llm_calls):
    """Test researcher node behavior when all scraping attempts fail."""
    mock_vertex, mock_openai = mock_llm_calls
    mock_openai.ainvoke.reset_mock()
    
    mock_response_1 = MagicMock()
    mock_response_1.content = '{"variants": ["quantum error correction codes", "surface codes stability"]}'
    mock_response_1.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    mock_openai.ainvoke.side_effect = []
    
    mock_search_results = [
        SearchResult(title="Surface Codes Guide", url="https://quantum-example.com/surface", content="Surface code tutorial.")
    ]
    
    mock_embeddings_instance = MagicMock()
    mock_embeddings_instance.aembed_documents = AsyncMock(return_value=[[0.1]*768])
    
    with patch("core.nodes.research.search_searxng", new_callable=AsyncMock) as mock_searxng, \
         patch("core.nodes.research.BrowserExplorer.explore_url", new_callable=AsyncMock) as mock_explore, \
         patch("core.nodes.research.get_embeddings", return_value=mock_embeddings_instance), \
         patch("core.nodes.research.PCTSEngine.search", new_callable=AsyncMock) as mock_mcts, \
         patch("core.nodes.research.evaluate_scraped_relevance", new_callable=AsyncMock) as mock_eval, \
         patch("core.nodes.research.MangoRouter.select_url") as mock_select, \
         patch("core.nodes.research.MCPHub") as mock_hub_class:
         
        mock_searxng.return_value = mock_search_results
        mock_mcts.return_value = ("mock intent", ["quantum error correction codes", "surface codes stability"])
        mock_eval.return_value = 0.0
        mock_select.side_effect = ["https://quantum-example.com/surface", None]
        
        mock_hub = MagicMock()
        mock_hub.connect_all = AsyncMock()
        mock_hub.shutdown = AsyncMock()
        mock_hub_class.return_value = mock_hub
        
        # Setup mock explore failures
        mock_explore.return_value = {"success": False, "error_message": "HTTP 403 Forbidden"}
        
        # Execute node
        result = await researcher_node(sample_researcher_input)
        
        # Verify failure states
        assert len(result["sub_questions_state"]) == 1
        updated_sub_q = result["sub_questions_state"][0]
        assert updated_sub_q.status == "failed"
        assert updated_sub_q.results_summary == "Failed to scrape any valid or accessible source content to answer this sub-question."
        
        assert len(result["verified_sources"]) == 1
        assert result["verified_sources"][0].accessible is False
        assert "HTTP 403 Forbidden" in result["verified_sources"][0].error_message
        
        assert len(result["errors"]) == 1
        assert "Scraping failed" in result["errors"][0]
