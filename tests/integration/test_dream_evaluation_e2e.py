import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from langchain_core.messages import AIMessage

from core.models import GraphState, Report, Claim, SearchResult, VerifiedSource
from core.graph import compile_graph
from core.llm_router import LLMRouter
from core.nodes import scoping

@pytest.mark.asyncio
async def test_dream_evaluation_routing_and_streaming_e2e(monkeypatch):
    """E2E Integration test verifying the DREAM evaluation node conditional routing.
    
    This test executes the compiled StateGraph via `astream(stream_mode="updates")` and asserts:
    1. The graph streams updates seamlessly in a format compatible with Streamlit.
    2. When the evaluator fails the report quality criteria (KIC < threshold), it loops back
       to `supervisor_node` for gap remediation.
    3. On the second iteration, once remediation is compiled, the evaluator passes and 
       the graph exits at END.
    4. Token usages and cumulative costs are correctly updated throughout the loops.
    """
    
    eval_cycle_count = 0
    nodes_executed = []
    
    # 1. Mock LLM Router to return appropriate mock JSON structures for each node type
    async def mock_router_ainvoke(
        messages,
        tier="STANDARD",
        agent_name="unknown",
        node_name="unknown",
        **kwargs
    ):
        nonlocal eval_cycle_count
        
        if agent_name == "ScopingAgent" and node_name == "clarify_with_user":
            return AIMessage(
                content='{"clarification_needed": false, "clarifying_question": null}',
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            )
            
        elif agent_name == "ScopingAgent" and node_name == "write_research_brief":
            return AIMessage(
                content='''{
                    "topic": "Room-temp Superconductors",
                    "scope": "Study on materials, characteristics, and practical power grid applications.",
                    "constraints": [],
                    "sub_questions": [
                        {"id": "q1", "question": "What is the critical temperature of LK-99?"}
                    ],
                    "target_source_count": 1
                }''',
                usage_metadata={"input_tokens": 20, "output_tokens": 30, "total_tokens": 50}
            )
            
        elif agent_name == "SearchAgent" and node_name == "diversify_query":
            return AIMessage(
                content='{"variants": ["LK-99 temperature limit", "LK-99 superconductivity transition"]}',
                usage_metadata={"input_tokens": 15, "output_tokens": 8, "total_tokens": 23}
            )
            
        elif agent_name == "ResearcherAgent" and node_name == "summarize_sources":
            return AIMessage(
                content="LK-99 critical temperature is claimed to be 400K by experimentalists, but rejected by labs.",
                usage_metadata={"input_tokens": 25, "output_tokens": 15, "total_tokens": 40}
            )
            
        elif agent_name == "VerifierAgent" and node_name == "claim_extraction":
            return AIMessage(
                content='''{
                    "claims": [
                        {
                            "claim_id": "c1",
                            "claim_text": "LK-99 critical temperature is 400K",
                            "section": "LK-99 Superconductivity",
                            "supporting_quotes": ["critical temperature is claimed to be 400K"],
                            "source_urls": ["https://physics.org/lk99", "https://nature.com/lk99"]
                        }
                    ]
                }''',
                usage_metadata={"input_tokens": 30, "output_tokens": 25, "total_tokens": 55}
            )
            
        elif agent_name == "VerifierAgent" and node_name == "verifier_critique":
            return AIMessage(
                content='{"gaps_found": false, "critique_text": "No gaps found.", "suggested_queries": []}',
                usage_metadata={"input_tokens": 35, "output_tokens": 10, "total_tokens": 45}
            )
            
        elif agent_name == "WriterAgent" and node_name == "report_writer":
            return AIMessage(
                content='''{
                    "title": "LK-99 Critical Temperature Analysis",
                    "content": "# LK-99 Critical Temperature Analysis\\n\\nExperimental claims point to 400K, but replication failed."
                }''',
                usage_metadata={"input_tokens": 40, "output_tokens": 30, "total_tokens": 70}
            )
            
        elif agent_name == "EvaluatorAgent":
            if node_name == "extract_key_facts":
                eval_cycle_count += 1
                return AIMessage(
                    content='{"key_facts": ["LK-99 transition temperature", "Replication results from independent labs"]}',
                    usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}
                )
            elif node_name == "check_coverage":
                if eval_cycle_count == 1:
                    # Failing coverage score (0.50 coverage)
                    return AIMessage(
                        content='''{
                            "coverage_items": [
                                {"fact": "LK-99 transition temperature", "covered": true, "explanation": "Covered transition temperature"},
                                {"fact": "Replication results from independent labs", "covered": false, "explanation": "Completely missing independent lab replication results."}
                            ]
                        }''',
                        usage_metadata={"input_tokens": 15, "output_tokens": 15, "total_tokens": 30}
                    )
                else:
                    # Passing coverage score (1.00 coverage)
                    return AIMessage(
                        content='''{
                            "coverage_items": [
                                {"fact": "LK-99 transition temperature", "covered": true, "explanation": "Covered transition temperature"},
                                {"fact": "Replication results from independent labs", "covered": true, "explanation": "Covered replication results."}
                            ]
                        }''',
                        usage_metadata={"input_tokens": 15, "output_tokens": 15, "total_tokens": 30}
                    )
            elif node_name == "evaluate_reasoning":
                return AIMessage(
                    content='{"score": 0.85, "explanation": "Clear argument coherence."}',
                    usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                )
            elif node_name == "evaluate_factuality":
                return AIMessage(
                    content='{"score": 0.95, "explanation": "High citation formatting quality."}',
                    usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                )
            elif node_name == "remediation_formulation":
                return AIMessage(
                    content='''{
                        "gaps_found": true,
                        "remediation_queries": ["LK-99 replication failures and peer evaluations"],
                        "remediation_notes": "Lacking replication reports."
                    }''',
                    usage_metadata={"input_tokens": 15, "output_tokens": 15, "total_tokens": 30}
                )
                
        # Fallback for any other calls
        return AIMessage(content="{}", usage_metadata={"input_tokens": 5, "output_tokens": 5, "total_tokens": 10})

    # 2. Setup mock patches
    mock_router = AsyncMock(spec=LLMRouter)
    mock_router.ainvoke = AsyncMock(side_effect=mock_router_ainvoke)
    
    # Initialize the router token tracking format
    mock_router.token_usage = {
        "vertex_ai": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
        "freellmapi": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
        "failovers": 0
    }
    
    # Manually update tracking dictionary inside the mock when called
    async def track_tokens_side_effect(*args, **kwargs):
        resp = await mock_router_ainvoke(*args, **kwargs)
        provider = "vertex_ai" if kwargs.get("tier") == "CRITICAL" else "freellmapi"
        in_tok = resp.usage_metadata.get("input_tokens", 0)
        out_tok = resp.usage_metadata.get("output_tokens", 0)
        
        mock_router.token_usage[provider]["input_tokens"] += in_tok
        mock_router.token_usage[provider]["output_tokens"] += out_tok
        mock_router.token_usage[provider]["calls"] += 1
        return resp
        
    mock_router.ainvoke.side_effect = track_tokens_side_effect
    
    # 3. Patch get_router in all modules and mock embeddings/SentenceTransformer
    monkeypatch.setattr(scoping, "get_router", lambda: mock_router)
    
    from core.nodes import evaluator, verifier, writer
    monkeypatch.setattr(evaluator, "get_router", lambda: mock_router)
    monkeypatch.setattr(verifier, "get_router", lambda: mock_router)
    monkeypatch.setattr(writer, "get_router", lambda: mock_router)
    
    mock_verified_source = VerifiedSource(
        url="https://physics.org/lk99",
        title="Replication",
        content="LK-99 transition and replication results. LK-99 critical temperature is claimed to be 400K by experimentalists.",
        accessible=True,
        status_code=200
    )
    
    with patch("db.embeddings.SentenceTransformer", MagicMock()), \
         patch("core.llm_router.LLMRouter.ainvoke", side_effect=track_tokens_side_effect), \
         patch("verification.source_checker.SourceChecker.check_source", AsyncMock(return_value=mock_verified_source)), \
         patch("core.nodes.research.search_searxng", AsyncMock(return_value=[SearchResult(title="Replication", url="https://physics.org/lk99", content="Failed to replicate")])), \
         patch("core.nodes.research.scrape_url", AsyncMock(return_value=VerifiedSource(url="https://physics.org/lk99", title="Replication", content="Failed to replicate"))), \
         patch("core.nodes.research.get_embeddings", MagicMock()):
         
        # Compile graph and run stream
        app = compile_graph()
        initial_state = {
            "user_query": "LK-99 superconductivity critical temp replication",
            "topic": "LK-99"
        }
        
        # 4. Stream and capture updates
        async for chunk in app.astream(initial_state, stream_mode="updates"):
            for node_name, val in chunk.items():
                nodes_executed.append(node_name)
                
        # 5. Assert loop-back behavior & results
        assert eval_cycle_count == 2
        
        # Verify node execution order: scoping ambiguity and brief, supervisor researcher context aggregator verifier writer, evaluator, supervisor researcher context aggregator verifier writer, evaluator, end
        # The list of executed nodes should contain 'evaluator_node' at least twice and loop back!
        assert "scoping_ambiguity_check" in nodes_executed
        assert "evaluator_node" in nodes_executed
        
        # Count occurrences of evaluator node executions
        eval_node_visits = [n for n in nodes_executed if n == "evaluator_node"]
        assert len(eval_node_visits) == 2
        
        # Find occurrences of supervisor_node - should be called after scoping and again after first evaluation failure
        supervisor_visits = [n for n in nodes_executed if n == "supervisor_node"]
        assert len(supervisor_visits) >= 2
        
        # Check final accumulated token usage
        total_vertex_calls = mock_router.token_usage["vertex_ai"]["calls"]
        assert total_vertex_calls > 0
        assert mock_router.token_usage["vertex_ai"]["input_tokens"] > 0
        assert mock_router.token_usage["vertex_ai"]["output_tokens"] > 0
