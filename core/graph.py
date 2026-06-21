from typing import Any, Dict, Literal
from langgraph.graph import StateGraph, START, END
from core.models import GraphState
from core.nodes.scoping import clarify_with_user_node, write_research_brief_node

async def supervisor_placeholder(state: GraphState) -> Dict[str, Any]:
    """Placeholder node for the Supervisor Router in Phase 1."""
    return {}

def route_after_ambiguity_check(state: GraphState) -> Literal["write_research_brief", "__end__"]:
    """Routing function to determine if we should stop for clarification or compile the brief."""
    if state.clarification_needed:
        return "__end__"
    return "write_research_brief"

def compile_graph():
    """Build and compile the LangGraph StateGraph workflow."""
    workflow = StateGraph(GraphState)
    
    # Register graph nodes
    workflow.add_node("scoping_ambiguity_check", clarify_with_user_node)
    workflow.add_node("write_research_brief", write_research_brief_node)
    workflow.add_node("supervisor_placeholder", supervisor_placeholder)
    
    # Configure edges
    workflow.add_edge(START, "scoping_ambiguity_check")
    
    # Add conditional route logic after query check
    workflow.add_conditional_edges(
        "scoping_ambiguity_check",
        route_after_ambiguity_check,
        {
            "write_research_brief": "write_research_brief",
            "__end__": END
        }
    )
    
    workflow.add_edge("write_research_brief", "supervisor_placeholder")
    workflow.add_edge("supervisor_placeholder", END)
    
    return workflow.compile()
