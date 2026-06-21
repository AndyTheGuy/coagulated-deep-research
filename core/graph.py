from typing import Literal
import structlog
from langgraph.graph import StateGraph, START, END
from core.models import GraphState
from core.nodes import (
    clarify_with_user_node, 
    write_research_brief_node, 
    researcher_node, 
    context_aggregator_node,
    verifier_node,
    report_writer_node
)
from core.router import supervisor_node, route_research

logger = structlog.get_logger("deep-research")

def route_after_ambiguity_check(state: GraphState) -> Literal["write_research_brief", "__end__"]:
    """Routing function to determine if we should stop for clarification or compile the brief."""
    if state.clarification_needed:
        return "__end__"
    return "write_research_brief"

def route_after_verification(state: GraphState) -> Literal["supervisor_node", "report_writer_node"]:
    """Conditional routing function after verification critique.
    If there are any pending sub-questions (added as gap-filling by the verifier),
    route back to the supervisor to spawn researcher subagents.
    Otherwise, proceed to the report writer.
    """
    logger.info("Routing after verification critique")
    sub_questions = state.sub_questions_state or []
    
    pending_questions = [q for q in sub_questions if q.status == "pending"]
    if pending_questions:
        logger.info("Verifier found gaps! Routing back to supervisor for gap-filling research", count=len(pending_questions))
        return "supervisor_node"
        
    logger.info("No gaps found. Routing to report writer to compile final report.")
    return "report_writer_node"

def compile_graph():
    """Build and compile the LangGraph StateGraph workflow for Phase 3."""
    workflow = StateGraph(GraphState)
    
    # Register graph nodes
    workflow.add_node("scoping_ambiguity_check", clarify_with_user_node)
    workflow.add_node("write_research_brief", write_research_brief_node)
    workflow.add_node("supervisor_node", supervisor_node)
    workflow.add_node("researcher_node", researcher_node)
    workflow.add_node("context_aggregator", context_aggregator_node)
    workflow.add_node("verifier_node", verifier_node)
    workflow.add_node("report_writer_node", report_writer_node)
    
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
    
    # Route aggregator output to verification pipeline instead of ending
    workflow.add_edge("context_aggregator", "verifier_node")
    
    # Handle self-correction feedback loop after verification
    workflow.add_conditional_edges(
        "verifier_node",
        route_after_verification,
        {
            "supervisor_node": "supervisor_node",
            "report_writer_node": "report_writer_node"
        }
    )
    
    # Final report is produced by report writer node
    workflow.add_edge("report_writer_node", END)
    
    return workflow.compile()

