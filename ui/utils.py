from typing import Any, Dict
from core.models import GraphState, reduce_sub_questions, merge_dict_reducer

def update_graph_state_with_chunk(current_state: GraphState, chunk: Dict[str, Any]) -> None:
    """Merge streamed node state dictionaries into the provided GraphState instance.
    
    This function processes LangGraph updates and correctly applies the annotated
    state reducers (e.g. lists addition, dict merging) to match LangGraph state management.
    """
    for node_name, node_update in chunk.items():
        if not isinstance(node_update, dict):
            continue
        for key, value in node_update.items():
            if value is None:
                continue
            if hasattr(current_state, key):
                curr_val = getattr(current_state, key)
                if key == "sub_questions_state":
                    setattr(current_state, key, reduce_sub_questions(curr_val, value))
                elif key in ["search_results", "verified_sources", "claims", "errors", "logs"]:
                    if curr_val is None:
                        curr_val = []
                    setattr(current_state, key, curr_val + value)
                elif key == "token_usage":
                    setattr(current_state, key, merge_dict_reducer(curr_val, value))
                else:
                    setattr(current_state, key, value)

def update_cost_stats_from_state(current_state: GraphState, cost_stats: Dict[str, Any]) -> None:
    """Pull token usage statistics from graph state and compute financial costs.
    
    Uses 2026 pricing:
    - Vertex AI (Gemini 3.5 Flash): Input = $0.0375 / 1M, Output = $0.15 / 1M
    - FreeLLMAPI: Cost = $0.00
    """
    token_usage = getattr(current_state, "token_usage", {}) or {}
    
    vertex = token_usage.get("vertex_ai", {"input_tokens": 0, "output_tokens": 0, "calls": 0})
    freellm = token_usage.get("freellmapi", {"input_tokens": 0, "output_tokens": 0, "calls": 0})
    failovers = token_usage.get("failovers", 0)
    
    cost_stats["vertex_calls"] = vertex.get("calls", 0)
    cost_stats["vertex_input"] = vertex.get("input_tokens", 0)
    cost_stats["vertex_output"] = vertex.get("output_tokens", 0)
    
    cost_stats["vertex_cost"] = (
        (vertex.get("input_tokens", 0) * 0.0375 / 1000000) +
        (vertex.get("output_tokens", 0) * 0.15 / 1000000)
    )
    
    cost_stats["freellm_calls"] = freellm.get("calls", 0)
    cost_stats["freellm_input"] = freellm.get("input_tokens", 0)
    cost_stats["freellm_output"] = freellm.get("output_tokens", 0)
    cost_stats["failovers"] = failovers

def estimate_remaining_cost(current_state: GraphState, cost_stats: Dict[str, Any]) -> float:
    """Compute an estimated remaining cost based on pending/in_progress sub-questions count."""
    sub_questions = getattr(current_state, "sub_questions_state", []) or []
    if not sub_questions:
        return 0.0015
    
    completed_count = sum(1 for q in sub_questions if q.status in ["completed", "failed"])
    remaining_count = sum(1 for q in sub_questions if q.status in ["pending", "in_progress"])
    
    current_cost = cost_stats.get("vertex_cost", 0.0)
    
    if completed_count > 0:
        cost_per_sub = current_cost / completed_count
        return cost_per_sub * remaining_count
    else:
        return remaining_count * 0.0005
