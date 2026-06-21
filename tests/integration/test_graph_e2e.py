import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from langchain_core.messages import AIMessage

from core.models import GraphState, Report, Claim, SearchResult, VerifiedSource
from core.graph import compile_graph
from core.nodes import scoping

@pytest.mark.asyncio
async def test_compiled_graph_specific_query_run(monkeypatch):
    """Test that the compiled graph completes scoping and transitions successfully 
    through research, verification, and report compilation completely offline.
    """
    # 1. Define E2E Mock LLM Router Side-Effect Function
    async def mock_router_ainvoke(
        messages,
        tier="STANDARD",
        agent_name="unknown",
        node_name="unknown",
        **kwargs
    ):
        if agent_name == "ScopingAgent" and node_name == "clarify_with_user":
            return AIMessage(
                content='{"clarification_needed": false, "clarifying_question": null}',
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            )
        elif agent_name == "ScopingAgent" and node_name == "write_research_brief":
            return AIMessage(
                content='''{
                    "topic": "Quantum Key Distribution",
                    "scope": "Detailed scope of QKD",
                    "constraints": ["use local papers"],
                    "sub_questions": [
                        {"id": "q1", "question": "What is QKD security?"}
                    ],
                    "target_source_count": 20
                }''',
                usage_metadata={"input_tokens": 15, "output_tokens": 15, "total_tokens": 30}
            )
        elif agent_name == "SearchAgent" and node_name == "diversify_query":
            return AIMessage(
                content='{"variants": ["QKD security principles"]}',
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            )
        elif agent_name == "ResearcherAgent" and node_name == "summarize_sources":
            return AIMessage(
                content="Quantum Key Distribution provides security based on physics rules.",
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            )
        elif agent_name == "VerifierAgent" and node_name == "claim_extraction":
            return AIMessage(
                content='''{
                    "claims": [
                        {
                            "claim_id": "c1",
                            "claim_text": "Quantum Key Distribution provides security based on physics rules",
                            "section": "Q1: What is QKD security?",
                            "supporting_quotes": ["physics rules"],
                            "source_urls": ["https://quantum.org/qkd"]
                        }
                    ]
                }''',
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            )
        elif agent_name == "VerifierAgent" and node_name == "verifier_critique":
            return AIMessage(
                content='{"gaps_found": false, "critique_text": "No gaps found.", "suggested_queries": []}',
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            )
        elif agent_name == "WriterAgent" and node_name == "report_writer":
            return AIMessage(
                content='''{
                    "title": "Quantum Key Distribution Report",
                    "content": "# Quantum Key Distribution Report\\n\\nSecurity is based on physics rules.",
                    "confidence_score": 0.7
                }''',
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            )
        return AIMessage(content="{}", usage_metadata={"input_tokens": 5, "output_tokens": 5, "total_tokens": 10})

    # 2. Monkeypatch the router singleton inside scoping
    monkeypatch.setattr(scoping.router, "ainvoke", mock_router_ainvoke)

    # 3. Setup Mocks for Search, Scraping, Embedding, and Source Verification
    mock_search_results = [
        SearchResult(
            title="QKD Source",
            url="https://quantum.org/qkd",
            content="Quantum Key Distribution provides security based on physics rules."
        )
    ]
    
    mock_scrape_result = (
        "QKD Source",
        "Quantum Key Distribution provides security based on physics rules."
    )
    
    mock_verified_source = VerifiedSource(
        url="https://quantum.org/qkd",
        title="QKD Source",
        content="Quantum Key Distribution provides security based on physics rules.",
        accessible=True,
        status_code=200
    )

    mock_embeddings_instance = MagicMock()
    mock_embeddings_instance.embed_documents = MagicMock(return_value=[[1.0] + [0.0]*767])
    mock_embeddings_instance.embed_query = MagicMock(return_value=[1.0] + [0.0]*767)
    mock_embeddings_instance.aembed_documents = AsyncMock(return_value=[[1.0] + [0.0]*767])
    mock_embeddings_instance.aembed_query = AsyncMock(return_value=[1.0] + [0.0]*767)

    # Patch modules to run 100% offline
    with patch("db.embeddings.SentenceTransformer") as mock_st_class, \
         patch("core.llm_router.LLMRouter.ainvoke", side_effect=mock_router_ainvoke) as mock_local_router, \
         patch("core.nodes.research.search_searxng", new_callable=AsyncMock) as mock_searxng, \
         patch("core.nodes.research.search_ddg", new_callable=AsyncMock) as mock_ddg, \
         patch("core.nodes.research.scrape_url", new_callable=AsyncMock) as mock_scrape, \
         patch("core.nodes.research.get_embeddings", return_value=mock_embeddings_instance), \
         patch("verification.source_checker.SourceChecker.check_source", new_callable=AsyncMock) as mock_check_source:
         
        mock_st_class.return_value = MagicMock()
        mock_searxng.return_value = mock_search_results
        mock_ddg.return_value = []
        mock_scrape.return_value = mock_scrape_result
        mock_check_source.return_value = mock_verified_source

        # 4. Compile the graph
        app = compile_graph()

        # 5. Invoke graph
        initial_state = {
            "user_query": "Explain how Quantum Key Distribution (QKD) works.",
            "topic": "Quantum Key Distribution"
        }
        final_state = await app.ainvoke(initial_state)

        # 6. Assertions
        assert final_state["clarification_needed"] is False
        assert final_state["research_brief"] is not None
        assert final_state["research_brief"].topic == "Quantum Key Distribution"
        assert len(final_state["sub_questions_state"]) == 1
        assert final_state["sub_questions_state"][0].id == "q1"


@pytest.mark.asyncio
async def test_compiled_graph_ambiguous_query_run(monkeypatch):
    """Test that the compiled graph exits early for clarification on an ambiguous query."""
    # 1. Compile the graph
    app = compile_graph()
    
    # 2. Mock the LLM router
    mock_ainvoke = AsyncMock()
    mock_ainvoke.side_effect = [
        AsyncMock(
            content='{"clarification_needed": true, "clarifying_question": "What specific aspect of Quantum Computing?"}',
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        )
    ]
    monkeypatch.setattr(scoping.router, "ainvoke", mock_ainvoke)
    
    # 3. Invoke graph
    initial_state = {
        "user_query": "Quantum Computing",
        "topic": "Quantum Computing"
    }
    final_state = await app.ainvoke(initial_state)
    
    assert final_state["clarification_needed"] is True
    assert final_state["clarification_question"] == "What specific aspect of Quantum Computing?"
    assert final_state.get("research_brief") is None
