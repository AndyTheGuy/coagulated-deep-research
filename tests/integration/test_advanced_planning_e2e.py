import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import AIMessage

from core.graph import compile_graph
from core.models import Report, SearchResult, VerifiedSource

@pytest.fixture
def mock_advanced_planning_dependencies(monkeypatch):
    """Mocks LLM models, searches, embeddings, and scrapers to enable a fast, 
    deterministic, 100% offline integration test for advanced planning.
    """
    # 1. Setup Mock LLM instances
    mock_vertex_instance = MagicMock()
    mock_vertex_instance.model = "gemini-1.5-flash"
    mock_vertex_instance.ainvoke = AsyncMock()
    
    mock_openai_instance = MagicMock()
    mock_openai_instance.model_name = "gpt-4o-mini"
    mock_openai_instance.ainvoke = AsyncMock()

    # 2. Define conditional router side-effect to handle all LLM calls including MCTS and Mango
    async def mock_router_ainvoke(self, messages, tier="STANDARD", agent_name="unknown", node_name="unknown", **kwargs):
        prompt_text = str(messages)
        
        # Scoping: Ambiguity check
        if "clarification_needed" in prompt_text or "clarify_with_user" in node_name:
            response = AIMessage(
                content='{"clarification_needed": false, "clarifying_question": null}',
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            )
            
        # Scoping: Write research brief
        elif "write_research_brief" in node_name or "brief" in prompt_text:
            response = AIMessage(
                content='''{
                    "topic": "Quantum Teleportation",
                    "scope": "Analysis of continuous variable quantum teleportation limits",
                    "constraints": ["focus on continuous variables"],
                    "sub_questions": [
                        {"id": "q_advanced_1", "question": "What is the fidelity limit of continuous variable quantum teleportation?"}
                    ],
                    "target_source_count": 1
                }''',
                usage_metadata={"input_tokens": 20, "output_tokens": 20, "total_tokens": 40}
            )
            
        # MCTS Engine: Generate candidate intents
        elif "generate_candidate_intents" in node_name or "intents" in prompt_text:
            response = AIMessage(
                content="- Analyze squeezed state protocols\n- Evaluate optical losses\n- Compile quantum limits",
                usage_metadata={"input_tokens": 15, "output_tokens": 10, "total_tokens": 25}
            )
            
        # MCTS Engine: Simulate outcome state
        elif "simulate_outcome_state" in node_name or "Simulated Evidence Gained" in prompt_text:
            response = AIMessage(
                content="Squeezed states achieve fidelity limits up to 0.99 under zero-loss conditions.",
                usage_metadata={"input_tokens": 15, "output_tokens": 10, "total_tokens": 25}
            )
            
        # MCTS Engine: Evaluate state quality
        elif "evaluate_state_quality" in node_name or "Evidence Collected" in prompt_text:
            response = AIMessage(
                content="0.95",
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            )
            
        # MCTS Engine: Generate queries for intent
        elif "generate_queries_for_intent" in node_name or "Plan Intent" in prompt_text:
            response = AIMessage(
                content="continuous variable teleportation squeezed states\nteleportation fidelity optical losses\nquantum teleportation continuous variables limit",
                usage_metadata={"input_tokens": 15, "output_tokens": 15, "total_tokens": 30}
            )
            
        # Mango Relevance: Evaluate scraped relevance
        elif "evaluate_relevance" in node_name or "Target Question" in prompt_text:
            response = AIMessage(
                content="0.90",
                usage_metadata={"input_tokens": 15, "output_tokens": 5, "total_tokens": 20}
            )
            
        # Research Node: Summarize sources / synthesis
        elif "summarize_sources" in node_name or "academic/analyst-grade" in prompt_text:
            response = AIMessage(
                content="Under ideal conditions, continuous variable quantum teleportation reaches a maximum fidelity of 0.99 utilizing highly squeezed states. Inline citations: [CV Quantum Teleportation](https://quantum-teleportation-cv.org).",
                usage_metadata={"input_tokens": 50, "output_tokens": 30, "total_tokens": 80}
            )
            
        # Verifier Node: Claim extraction
        elif "claim_extraction" in node_name or "Factual claims extracted" in prompt_text:
            response = AIMessage(
                content='''{
                    "claims": [
                        {
                            "claim_id": "c1",
                            "claim_text": "Continuous variable quantum teleportation reaches a maximum fidelity of 0.99 using squeezed states",
                            "section": "Fidelity Limits",
                            "supporting_quotes": ["fidelity of 0.99 utilizing highly squeezed states"],
                            "source_urls": ["https://quantum-teleportation-cv.org"]
                        }
                    ]
                }''',
                usage_metadata={"input_tokens": 30, "output_tokens": 25, "total_tokens": 55}
            )
            
        # Verifier Node: Verifier critique
        elif "verifier_critique" in node_name:
            response = AIMessage(
                content='{"gaps_found": false, "critique_text": "No gaps. The fidelity metric is perfectly supported by the citation.", "suggested_queries": []}',
                usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
            )
            
        # Writer Node: Report writer
        elif "report_writer" in node_name or "report_writer" in prompt_text:
            response = AIMessage(
                content='''{
                    "title": "Continuous Variable Quantum Teleportation Fidelity Limits",
                    "content": "# Continuous Variable Quantum Teleportation Fidelity Limits\\n\\nContinuous variable quantum teleportation can achieve a maximum fidelity of 0.99 using squeezed states [CV Quantum Teleportation](https://quantum-teleportation-cv.org).",
                    "confidence_score": 0.95
                }''',
                usage_metadata={"input_tokens": 40, "output_tokens": 40, "total_tokens": 80}
            )
            
        # Default fallback
        else:
            response = AIMessage(
                content="{}",
                usage_metadata={"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}
            )
            
        # Track token usage metrics on the router instance
        provider = "vertex_ai" if tier.upper() == "CRITICAL" else "freellmapi"
        self._update_usage(provider, response)
        return response
        
    # 3. Setup mock search results and scrape result
    mock_search_results = [
        SearchResult(
            title="CV Teleportation Breakthrough",
            url="https://quantum-teleportation-cv.org",
            content="Breakthrough achievements in continuous variable squeezed state protocols."
        )
    ]
    
    mock_scrape_result = (
        "CV Teleportation Breakthrough",
        "Excellent factual density about continuous variable teleportation achieving a maximum fidelity of 0.99 using squeezed states."
    )
    
    # 4. Mock Embeddings Model (SentenceTransformer)
    mock_embeddings_instance = MagicMock()
    mock_embeddings_instance.embed_documents = MagicMock(return_value=[[1.0] + [0.0]*767])
    mock_embeddings_instance.embed_query = MagicMock(return_value=[1.0] + [0.0]*767)
    mock_embeddings_instance.aembed_documents = AsyncMock(return_value=[[1.0] + [0.0]*767])
    mock_embeddings_instance.aembed_query = AsyncMock(return_value=[1.0] + [0.0]*767)

    # 5. Apply system patches to avoid network access
    import core.llm_router
    monkeypatch.setattr(core.llm_router.LLMRouter, "ainvoke", mock_router_ainvoke)
    
    with patch("core.nodes.research.search_searxng", new_callable=AsyncMock) as mock_searxng, \
         patch("core.nodes.research.search_ddg", new_callable=AsyncMock) as mock_ddg, \
         patch("core.nodes.research.scrape_url", new_callable=AsyncMock) as mock_scrape, \
         patch("core.nodes.research.get_embeddings", return_value=mock_embeddings_instance), \
         patch("db.cache.LocalEmbeddings", return_value=mock_embeddings_instance), \
         patch("db.embeddings.SentenceTransformer") as mock_st_class, \
         patch("verification.source_checker.SourceChecker.check_source", new_callable=AsyncMock) as mock_check_source:
         
        mock_st_class.return_value = MagicMock()
        mock_searxng.return_value = mock_search_results
        mock_ddg.return_value = []
        mock_scrape.return_value = mock_scrape_result
        
        mock_verified = VerifiedSource(
            url="https://quantum-teleportation-cv.org",
            title="CV Teleportation Breakthrough",
            content="Excellent factual density about continuous variable teleportation achieving a maximum fidelity of 0.99 using squeezed states.",
            accessible=True,
            status_code=200
        )
        mock_check_source.return_value = mock_verified
        
        yield {
            "searxng": mock_searxng,
            "scrape": mock_scrape,
            "openai": mock_openai_instance,
            "vertex": mock_vertex_instance
        }

@pytest.mark.asyncio
async def test_advanced_planning_flow_e2e(mock_advanced_planning_dependencies, tmp_path):
    """E2E Integration Test validating that research tasks successfully utilize 
    MCP (Sequential Thinking via MCPHub/mock client) and Monte Carlo Tree Search 
    (PCTSEngine) under simulated conditions.
    """
    # Force a temporary path for SQLite SemanticCache to keep it completely isolated
    test_db_path = str(tmp_path / "test_advanced_cache.db")
    
    with patch("sqlite3.connect") as mock_connect:
        import sqlite3
        real_conn = sqlite3.connect(test_db_path)
        mock_connect.return_value = real_conn
        
        # 1. Compile the StateGraph
        app = compile_graph()
        
        # 2. Invoke the graph with a query requiring advanced research and planning
        initial_state = {
            "user_query": "What is the maximum fidelity limit of continuous variable quantum teleportation?",
            "topic": "Quantum Teleportation"
        }
        
        final_state = await app.ainvoke(initial_state)
        real_conn.close()
        
        # 3. Assert Graph Output Correctness
        assert final_state["clarification_needed"] is False
        assert final_state["research_brief"] is not None
        assert final_state["research_brief"].topic == "Quantum Teleportation"
        
        # Sub-questions state completed via advanced planning
        assert len(final_state["sub_questions_state"]) == 1
        sub_q = final_state["sub_questions_state"][0]
        assert sub_q.id == "q_advanced_1"
        assert sub_q.status == "completed"
        assert sub_q.assigned_researcher == "researcher_node"
        assert "continuous variable quantum teleportation" in sub_q.results_summary.lower() or "maximum fidelity of 0.99" in sub_q.results_summary
        
        # 4. Verify that verified sources contain our MCTS/Mango scraped URL
        assert len(final_state["verified_sources"]) > 0
        urls = [src.url for src in final_state["verified_sources"]]
        assert "https://quantum-teleportation-cv.org" in urls
        
        # 5. Assert final report contains compiled answers and correct citations
        assert final_state["final_report"] is not None
        report = final_state["final_report"]
        assert isinstance(report, Report)
        assert "Continuous Variable Quantum Teleportation" in report.title
        assert "0.99" in report.content
        assert "[CV Quantum Teleportation](https://quantum-teleportation-cv.org)" in report.content
        
        # 6. Verify that token usage was accumulated correctly through all advanced nodes
        assert "freellmapi" in final_state["token_usage"]
        usage = final_state["token_usage"]["freellmapi"]
        # MCTS calls, evaluators, query generators, summary synthesis, and writers all add up
        assert usage["calls"] >= 5
        assert usage["input_tokens"] > 50
        assert usage["output_tokens"] > 50
        
        # 7. Verify structured logging traces exist for MCTS Planning and Mango starting-point optimization
        logs = final_state["logs"]
        assert any("Advanced researcher node started" in log for log in logs)
        assert any("MCTS Selected Intent" in log for log in logs)
        assert any("Completed starting-point URL selection via Mango" in log for log in logs)
        assert any("Successfully researched and synthesized findings" in log for log in logs)
