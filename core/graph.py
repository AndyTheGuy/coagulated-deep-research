from typing import Literal
from langgraph.graph import StateGraph, START, END
from core.models import GraphState
from core.nodes import clarify_with_user_node, write_research_brief_node, researcher_node, context_aggregator_node
from core.router import supervisor_node, route_research

def route_after_ambiguity_check(state: GraphState) -> Literal["write_research_brief", "__end__"]:
    """Routing function to determine if we should stop for clarification or compile the brief."""
    if state.clarification_needed:
        return "__end__"
    return "write_research_brief"

def compile_graph():
    """Build and compile the LangGraph StateGraph workflow for Phase 2."""
    workflow = StateGraph(GraphState)
    
    # Register graph nodes
    workflow.add_node("scoping_ambiguity_check", clarify_with_user_node)
    workflow.add_node("write_research_brief", write_research_brief_node)
    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("researcher_node", researcher_node)
    workflow.add_node("context_aggregator", context_aggregator_node)
    
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
    
    # Research map-reduce loop
    workflow.add_edge("write_research_brief", "supervisor_node")
    
    workflow.add_conditional_edges(
        "supervisor_node",
        route_research,
        {
            "researcher_node": "researcher_node",
            "context_aggregator": "context_aggregator"
        }
    )
    
    # Researcher loops back to supervisor to verify progress and check for remaining tasks
    workflow.add_edge("researcher_node", "supervisor_node")
    
    # Compile and output
    workflow.add_edge("context_aggregator", END)
    
    return workflow.compile()
