import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core.graph import compile_graph
from core.models import Report, SearchResult

@pytest.fixture
def mock_pipeline_dependencies(monkeypatch):
    """Mocks all external models and services (LLMs, SearXNG, scrapers, embeddings)
    to enable a fast, deterministic, offline integration test.
    """
    # Mock LLM model instances
    mock_vertex_instance = MagicMock()
    mock_vertex_instance.model = "gemini-1.5-flash"
    mock_vertex_instance.ainvoke = AsyncMock()
    
    mock_openai_instance = MagicMock()
    mock_openai_instance.model_name = "gpt-4o-mini"
    mock_openai_instance.ainvoke = AsyncMock()
    
    # 1. Scoping stage 1: clarification_needed -> False
    res_scoping_1 = MagicMock()
    res_scoping_1.content = '{"clarification_needed": false, "clarifying_question": null}'
    res_scoping_1.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    
    # 2. Scoping stage 2: write_research_brief -> topic, scope, constraints, sub_questions
    res_scoping_2 = MagicMock()
    res_scoping_2.content = '{"topic": "Artificial General Intelligence", "scope": "Consensus on timelines", "constraints": ["focus on predictions"], "sub_questions": [{"id": "q1", "question": "What are AGI timeline consensus estimates?"}], "target_source_count": 2}'
    res_scoping_2.usage_metadata = {"input_tokens": 20, "output_tokens": 20, "total_tokens": 40}
    
    # 3. Research stage 1: query diversification -> variants
    res_diversify = MagicMock()
    res_diversify.content = '{"variants": ["AGI timelines consensus estimates", "artificial general intelligence timeframe"]}'
    res_diversify.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    
    # 4. Research stage 2: results summary synthesis -> summary text
    res_summarize = MagicMock()
    res_summarize.content = "Consensus estimates from leading research groups suggest AGI emergence between 2030 and 2050 with broad error margins."
    res_summarize.usage_metadata = {"input_tokens": 30, "output_tokens": 15, "total_tokens": 45}
    
    # Hook side effects
    mock_openai_instance.ainvoke.side_effect = [
        res_scoping_1,
        res_diversify,
        res_summarize
    ]
    mock_vertex_instance.ainvoke.return_value = res_scoping_2
    
    mock_vertex_cls = MagicMock(return_value=mock_vertex_instance)
    mock_openai_cls = MagicMock(return_value=mock_openai_instance)
    
    monkeypatch.setattr("core.llm_router.ChatVertexAI", mock_vertex_cls)
    monkeypatch.setattr("core.llm_router.ChatOpenAI", mock_openai_cls)
    
    # Reset scoping and diversify singletons to ensure they pick up our monkeypatched classes
    import core.nodes.scoping
    import search.diversify
    monkeypatch.setattr(core.nodes.scoping, "_router", None)
    monkeypatch.setattr(search.diversify, "_router", None)
    
    # Mock Search Result Sets
    mock_search_results = [
        SearchResult(title="AGI Forecast 2026", url="https://agi-forecast.org/2026", content="Expert projections on AGI development.")
    ]
    
    # Mock Embeddings Model (SentenceTransformer)
    mock_embeddings_instance = MagicMock()
    mock_embeddings_instance.aembed_documents = AsyncMock(return_value=[
        [1.0] + [0.0]*767,
        [0.0] + [1.0] + [0.0]*766
    ])
    mock_embeddings_instance.aembed_query = AsyncMock(return_value=[1.0] + [0.0]*767)
    
    # Apply patches
    with patch("core.nodes.research.search_searxng", new_callable=AsyncMock) as mock_searxng, \
         patch("core.nodes.research.scrape_url", new_callable=AsyncMock) as mock_scrape, \
         patch("core.nodes.research.get_embeddings", return_value=mock_embeddings_instance), \
         patch("db.cache.LocalEmbeddings", return_value=mock_embeddings_instance):
         
        mock_searxng.return_value = mock_search_results
        mock_scrape.return_value = ("AGI Forecast 2026", "Expert panels suggest AGI is highly probable in the coming decades.")
        
        yield {
            "searxng": mock_searxng,
            "scrape": mock_scrape,
            "openai": mock_openai_instance
        }

@pytest.mark.asyncio
async def test_full_search_and_aggregation_pipeline(mock_pipeline_dependencies, tmp_path):
    # Set custom database path for SemanticCache during test
    test_db_path = str(tmp_path / "test_cache.db")
    
    # Patch sqlite3.connect inside cache to always connect to our temporary test db
    with patch("sqlite3.connect") as mock_connect:
        import sqlite3
        real_conn = sqlite3.connect(test_db_path)
        mock_connect.return_value = real_conn
        
        # 1. Compile graph
        app = compile_graph()
        
        # 2. Invoke graph with an initial query
        initial_state = {
            "user_query": "When will AGI be created?",
        }
        
        final_state = await app.ainvoke(initial_state)
        
        # Close connection
        real_conn.close()
        
        # 3. Verify graph outputs
        assert final_state["clarification_needed"] is False
        assert final_state["research_brief"] is not None
        assert final_state["research_brief"].topic == "Artificial General Intelligence"
        
        # Sub-questions state transitioned to completed
        assert len(final_state["sub_questions_state"]) == 1
        sub_q = final_state["sub_questions_state"][0]
        assert sub_q.id == "q1"
        assert sub_q.status == "completed"
        assert sub_q.assigned_researcher == "researcher_node"
        assert "Consensus estimates from leading research groups" in sub_q.results_summary
        
        # Report generated successfully with content and citations
        assert final_state["draft_report"] is not None
        report = final_state["draft_report"]
        assert isinstance(report, Report)
        assert "Artificial General Intelligence" in report.title
        assert "Consensus estimates from leading research groups" in report.content
        assert "Sources & References" in report.content
        assert "[AGI Forecast 2026](https://agi-forecast.org/2026)" in report.content
        
        # Assert citations
        assert len(report.citations) == 1
        assert report.citations[0] == "https://agi-forecast.org/2026"
        
        # Assert accumulated token usage via custom dict reducer
        assert "freellmapi" in final_state["token_usage"]
        usage = final_state["token_usage"]["freellmapi"]
        assert usage["calls"] == 4
        assert usage["input_tokens"] == 60
        assert usage["output_tokens"] == 30
        
        assert "vertex_ai" in final_state["token_usage"]
        v_usage = final_state["token_usage"]["vertex_ai"]
        assert v_usage["calls"] == 2
        assert v_usage["input_tokens"] == 40
        assert v_usage["output_tokens"] == 40
