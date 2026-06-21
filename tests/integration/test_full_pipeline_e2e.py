import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from langchain_core.messages import AIMessage

from core.models import GraphState, Report, Claim, SearchResult, VerifiedSource
from core.graph import compile_graph
from core.llm_router import LLMRouter
from core.nodes import scoping

@pytest.mark.asyncio
async def test_full_pipeline_e2e(monkeypatch):
    """End-to-end integration test of the full StateGraph pipeline.
    
    This test verifies that a complete research query transitions successfully through
    all nodes (scoping, parallel researcher, context aggregator, verifier, and report writer)
    and terminates at END with a fully verified and formatted final report.
    """
    
    # 1. Define E2E Mock LLM Router Side-Effect Function
    async def mock_router_ainvoke(
        messages,
        tier="STANDARD",
        agent_name="unknown",
        node_name="unknown",
        **kwargs
    ):
        """Unified LLM router mock that returns tailored responses based on agent and node names."""
        
        if agent_name == "ScopingAgent" and node_name == "clarify_with_user":
            # No clarification required for this specific query run
            return AIMessage(
                content='{"clarification_needed": false, "clarifying_question": null}',
                usage_metadata={"input_tokens": 12, "output_tokens": 6, "total_tokens": 18}
            )
            
        elif agent_name == "ScopingAgent" and node_name == "write_research_brief":
            # Return research brief with one targeted sub-question
            return AIMessage(
                content='''{
                    "topic": "Quantum Cryptography",
                    "scope": "In-depth study of quantum cryptographic protocols, focusing on security and QKD.",
                    "constraints": ["focus on QKD"],
                    "sub_questions": [
                        {"id": "q1", "question": "What are the security principles of QKD?"}
                    ],
                    "target_source_count": 1
                }''',
                usage_metadata={"input_tokens": 20, "output_tokens": 30, "total_tokens": 50}
            )
            
        elif agent_name == "SearchAgent" and node_name == "diversify_query":
            # Diversify the sub-question into search query variants
            return AIMessage(
                content='{"variants": ["QKD security principles", "unbreakable quantum cryptography"]}',
                usage_metadata={"input_tokens": 15, "output_tokens": 8, "total_tokens": 23}
            )
            
        elif agent_name == "ResearcherAgent" and node_name == "summarize_sources":
            # Synthesize and answer the sub-question based on the scraped source
            return AIMessage(
                content="Quantum key distribution (QKD) relies on quantum mechanics to provide information-theoretic security. This is theoretically unbreakable as supported by [Quantum Example Source](https://quantum-example.com/qkd).",
                usage_metadata={"input_tokens": 25, "output_tokens": 18, "total_tokens": 43}
            )
            
        elif agent_name == "VerifierAgent" and node_name == "claim_extraction":
            # Extract claims from draft report for verification
            return AIMessage(
                content='''{
                    "claims": [
                        {
                            "claim_id": "c1",
                            "claim_text": "QKD provides information-theoretic security",
                            "section": "Q1: What are the security principles of QKD?",
                            "supporting_quotes": ["information-theoretic security"],
                            "source_urls": ["https://quantum-example.com/qkd"]
                        }
                    ]
                }''',
                usage_metadata={"input_tokens": 35, "output_tokens": 22, "total_tokens": 57}
            )
            
        elif agent_name == "VerifierAgent" and node_name == "verifier_critique":
            # Adversarial verifier critique - approve the draft report without gaps
            return AIMessage(
                content='{"gaps_found": false, "critique_text": "The draft report is comprehensive, and the claims are verified. No gaps found.", "suggested_queries": []}',
                usage_metadata={"input_tokens": 40, "output_tokens": 12, "total_tokens": 52}
            )
            
        elif agent_name == "WriterAgent" and node_name == "report_writer":
            # Final polished markdown report compilation
            return AIMessage(
                content='''{
                    "title": "Quantum Cryptography and QKD Security Analysis",
                    "content": "# Quantum Cryptography and QKD Security Analysis\\n\\nQuantum key distribution (QKD) relies on quantum mechanics to provide information-theoretic security. This is theoretically unbreakable.\\n\\n## Sources & References\\n- [Quantum Example Source](https://quantum-example.com/qkd)\\n\\n## Report Confidence Assessment\\nConfidence Score: 0.7"
                }''',
                usage_metadata={"input_tokens": 45, "output_tokens": 35, "total_tokens": 80}
            )
            
        else:
            return AIMessage(content="{}", usage_metadata={"input_tokens": 5, "output_tokens": 5, "total_tokens": 10})

    # 2. Monkeypatch the router singleton inside scoping to use our mock side-effect
    # (Scoping, verifier, writer, and claim extractor import scoping.router)
    monkeypatch.setattr(scoping.router, "ainvoke", mock_router_ainvoke)

    # 3. Setup Mocks for Search, Scraping, Embedding, and Source Verification
    mock_search_results = [
        SearchResult(
            title="Quantum Example Source",
            url="https://quantum-example.com/qkd",
            content="QKD relies on quantum mechanics to provide information-theoretic security."
        )
    ]
    
    mock_scrape_result = (
        "Quantum Example Source",
        "QKD relies on quantum mechanics to provide information-theoretic security."
    )
    
    # Mock source checker to return verified and accessible source document
    mock_verified_source = VerifiedSource(
        url="https://quantum-example.com/qkd",
        title="Quantum Example Source",
        content="QKD relies on quantum mechanics to provide information-theoretic security.",
        accessible=True,
        status_code=200
    )

    # Mock embeddings instance to bypass sentence-transformers model download
    mock_embeddings_instance = MagicMock()
    mock_embeddings_instance.embed_documents = MagicMock(return_value=[[1.0] + [0.0]*767])
    mock_embeddings_instance.embed_query = MagicMock(return_value=[1.0] + [0.0]*767)
    mock_embeddings_instance.aembed_documents = AsyncMock(return_value=[[1.0] + [0.0]*767])
    mock_embeddings_instance.aembed_query = AsyncMock(return_value=[1.0] + [0.0]*767)

    # Patch sentence-transformers loading in local embeddings globally to prevent downloading model
    with patch("db.embeddings.SentenceTransformer") as mock_st_class, \
         patch("core.llm_router.LLMRouter.ainvoke", side_effect=mock_router_ainvoke) as mock_local_router, \
         patch("core.nodes.research.search_searxng", new_callable=AsyncMock) as mock_searxng, \
         patch("core.nodes.research.search_ddg", new_callable=AsyncMock) as mock_ddg, \
         patch("core.nodes.research.scrape_url", new_callable=AsyncMock) as mock_scrape, \
         patch("core.nodes.research.get_embeddings", return_value=mock_embeddings_instance), \
         patch("verification.source_checker.SourceChecker.check_source", new_callable=AsyncMock) as mock_check_source:
         
        # Set up mock implementations
        mock_st_class.return_value = MagicMock()
        mock_searxng.return_value = mock_search_results
        mock_ddg.return_value = []
        mock_scrape.return_value = mock_scrape_result
        mock_check_source.return_value = mock_verified_source

        # 4. Compile the StateGraph
        app = compile_graph()

        # 5. Invoke graph with initial state
        initial_state = {
            "user_query": "Explain how Quantum Cryptography provides security.",
            "topic": "Quantum Cryptography"
        }
        
        final_state = await app.ainvoke(initial_state)

        # 6. E2E Assertions
        # Assertions on final_report structure and values
        assert final_state.get("final_report") is not None, "final_report must be populated"
        final_report = final_state["final_report"]
        assert isinstance(final_report, Report), "final_report must be an instance of Report"
        assert final_report.title == "Quantum Cryptography and QKD Security Analysis"
        assert "information-theoretic security" in final_report.content

        # Assertion: final_report.confidence_score matches verified claims
        # Score calculation: 1 verified source, exact quote matched = 0.5 + 0.2 * 1.0 = 0.70
        assert final_report.confidence_score == 0.70
        assert len(final_report.claims) == 1
        assert final_report.claims[0].claim_id == "c1"
        assert final_report.claims[0].verification_status == "verified"
        assert final_report.claims[0].confidence_score == 0.70

        # Assertions on sub-questions transitions and outcomes
        assert len(final_state["sub_questions_state"]) == 1
        sub_q = final_state["sub_questions_state"][0]
        assert sub_q.id == "q1"
        assert sub_q.status == "completed"
        assert sub_q.assigned_researcher == "researcher_node"

        # Assertions on logs containing compilation and verification summaries
        logs = final_state["logs"]
        assert len(logs) > 0
        
        # Verify scoping, aggregation, verification, and compilation logs
        assert any("Successfully researched and synthesized findings" in log for log in logs)
        assert any("Context aggregator successfully compiled" in log for log in logs)
        assert any("Verifier node: Verified report confidence score 0.7" in log for log in logs)
        assert any("Report writer compiled final report" in log for log in logs)
        
        # Verify token usages are tracked across nodes
        token_usage = final_state.get("token_usage", {})
        assert "freellmapi" in token_usage or "vertex_ai" in token_usage
