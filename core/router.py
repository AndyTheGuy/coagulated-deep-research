import structlog
from typing import Any, Dict, List, Literal, Union
from langgraph.types import Send
from core.models import GraphState, ResearcherInput

logger = structlog.get_logger("deep-research")

async def supervisor_node(state: GraphState) -> Dict[str, Any]:
    """Supervisor router node. Evaluates current research state, ensuring 
    sub_questions_state is populated from the research brief if starting out.
    """
    logger.info("Running supervisor_node", topic=state.topic)
    
    updates: Dict[str, Any] = {}
    
    # If sub_questions_state is empty but research_brief is present, initialize it
    if not state.sub_questions_state and state.research_brief:
        logger.info(
            "Initializing sub_questions_state from research_brief", 
            count=len(state.research_brief.sub_questions)
        )
        updates["sub_questions_state"] = state.research_brief.sub_questions
        
    return updates

def route_research(state: GraphState) -> Union[List[Send], Literal["context_aggregator"]]:
    """Conditional router function. Spawns parallel researchers for pending sub-questions, 
    or routes to the context aggregator if all sub-questions are complete.
    """
    logger.info("Running route_research decision")
    
    sub_questions = state.sub_questions_state
    if not sub_questions and state.research_brief:
        sub_questions = state.research_brief.sub_questions
        
    if not sub_questions:
        logger.warning("No sub-questions found to route, proceeding to context_aggregator")
        return "context_aggregator"
        
    pending_questions = [q for q in sub_questions if q.status == "pending"]
    
    if not pending_questions:
        logger.info("No pending sub-questions, proceeding to context_aggregator")
        return "context_aggregator"
        
    logger.info(
        "Spawning parallel researcher nodes", 
        pending_count=len(pending_questions),
        total_count=len(sub_questions)
    )
    
    sends = []
    for q in pending_questions:
        # Create the input payload for the parallel researcher node
        researcher_input = ResearcherInput(
            sub_question=q,
            topic=state.topic or (state.research_brief.topic if state.research_brief else ""),
            constraints=state.research_brief.constraints if state.research_brief else []
        )
        sends.append(Send("researcher_node", researcher_input))
        
    return sends
