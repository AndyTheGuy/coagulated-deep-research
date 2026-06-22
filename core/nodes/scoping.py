import asyncio
import contextvars
from typing import Any, Dict, Optional
import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from core.models import GraphState, ResearchBrief, SubQuestion
from core.llm_router import LLMRouter

logger = structlog.get_logger("deep-research")

_router_var: contextvars.ContextVar[Optional[LLMRouter]] = contextvars.ContextVar("scoping_router", default=None)
_router_loop_var: contextvars.ContextVar[Optional[asyncio.AbstractEventLoop]] = contextvars.ContextVar("scoping_router_loop", default=None)

# Compatibility global variable for tests that monkeypatch or reset '_router'
_router: Optional[LLMRouter] = None

def get_router() -> LLMRouter:
    global _router
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
        
    router_inst = _router_var.get()
    router_loop = _router_loop_var.get()
    
    # If _router was set to None externally (e.g., by a test via monkeypatch)
    if _router is None:
        router_inst = None
        
    if router_inst is None or router_loop != current_loop:
        router_inst = LLMRouter()
        _router_var.set(router_inst)
        _router_loop_var.set(current_loop)
        _router = router_inst
    return router_inst


class RouterProxy:
    """Proxy that defers LLMRouter instantiation until invocation, ensuring compatibility with unit tests."""
    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        return await get_router().ainvoke(*args, **kwargs)

router = RouterProxy()

class ClarificationOutput(BaseModel):
    """Pydantic model representing the output of the query ambiguity check."""
    clarification_needed: bool = Field(
        description="True if the query is too ambiguous, vague, or broad and needs user clarification; False otherwise"
    )
    clarifying_question: Optional[str] = Field(
        default=None,
        description="The clarifying question to ask the user if clarification_needed is True; None otherwise"
    )

async def clarify_with_user_node(state: GraphState) -> Dict[str, Any]:
    """Node that checks the user query for ambiguity and requests clarification if needed."""
    logger.info("Running clarify_with_user_node", query=state.user_query)
    
    # If the user has already provided a response to a clarification question, skip
    if state.clarification_response:
        logger.info("Clarification response found, skipping ambiguity detection")
        return {"clarification_needed": False}
        
    parser = JsonOutputParser(pydantic_object=ClarificationOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a research scoping assistant. Your job is to analyze the user's research query "
            "and determine if it is specific enough to conduct a high-quality deep research task.\n\n"
            "An ambiguous query is: too short (e.g. 'AI', 'quantum'), lacks context (e.g. 'what is the best model'), "
            "or covers too many different topics at once.\n\n"
            "Format the output strictly as a JSON object matching the following schema:\n"
            "{format_instructions}"
        )),
        ("user", "Analyze this query: {query}")
    ])
    
    formatted_prompt = prompt.format_prompt(
        query=state.user_query,
        format_instructions=parser.get_format_instructions()
    )
    
    response = await router.ainvoke(
        messages=formatted_prompt.to_messages(),
        tier="BULK",
        agent_name="ScopingAgent",
        node_name="clarify_with_user"
    )
    
    try:
        parsed = parser.parse(response.content)
        logger.info("Clarification analysis complete", clarification_needed=parsed.get("clarification_needed"))
        return {
            "clarification_needed": parsed.get("clarification_needed", False),
            "clarification_question": parsed.get("clarifying_question"),
            "token_usage": get_router().token_usage,
        }
    except Exception as e:
        logger.error("Failed to parse clarification output, defaulting to no clarification", error=str(e))
        return {
            "clarification_needed": False,
            "token_usage": get_router().token_usage,
        }

async def write_research_brief_node(state: GraphState) -> Dict[str, Any]:
    """Node that compiles the user query and any clarification into a structured ResearchBrief."""
    logger.info("Running write_research_brief_node", query=state.user_query)
    
    parser = JsonOutputParser(pydantic_object=ResearchBrief)
    
    # Combine query and clarification response if present
    full_input = f"Topic/Query: {state.user_query}"
    if state.clarification_question and state.clarification_response:
        full_input += f"\nClarifying Question asked: {state.clarification_question}\nUser Clarification: {state.clarification_response}"
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an academic/analyst-grade research architect. Your task is to compile a research brief "
            "based on the user's query and any clarification they provided.\n\n"
            "The brief must contain:\n"
            "- topic: A refined research topic title\n"
            "- scope: A detailed scope description specifying what is in-scope and out-of-scope\n"
            "- constraints: Core constraints (e.g. source requirements, timeframes)\n"
            "- sub_questions: A set of 3 to 6 logical, non-overlapping sub-questions that must be researched in parallel to cover the topic fully. Each must have an 'id' (e.g. 'q1', 'q2') and 'question' text.\n"
            "- target_source_count: Target source count (minimum 20 sources)\n\n"
            "Format the output strictly as a JSON object matching the following schema:\n"
            "{format_instructions}"
        )),
        ("user", "Generate research brief for:\n{input}")
    ])
    
    formatted_prompt = prompt.format_prompt(
        input=full_input,
        format_instructions=parser.get_format_instructions()
    )
    
    response = await router.ainvoke(
        messages=formatted_prompt.to_messages(),
        tier="CRITICAL",
        agent_name="ScopingAgent",
        node_name="write_research_brief"
    )
    
    try:
        parsed_brief = parser.parse(response.content)
        brief = ResearchBrief(**parsed_brief)
        logger.info("Generated research brief", topic=brief.topic, sub_questions=[q.question for q in brief.sub_questions])
        
        return {
            "research_brief": brief,
            "sub_questions_state": brief.sub_questions,
            "token_usage": get_router().token_usage,
        }
    except Exception as e:
        logger.error("Failed to parse or validate research brief", error=str(e))
        # Provide a fallback brief to prevent graph failure
        fallback_brief = ResearchBrief(
            topic=state.topic or state.user_query,
            scope="Fallback scope due to brief generation failure",
            sub_questions=[SubQuestion(id="q1", question=f"General research on {state.user_query}")],
            target_source_count=20
        )
        return {
            "research_brief": fallback_brief,
            "sub_questions_state": fallback_brief.sub_questions,
            "errors": [f"Scoping failed: {str(e)}"],
            "token_usage": get_router().token_usage,
        }
