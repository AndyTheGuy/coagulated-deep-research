import asyncio
import contextvars
from typing import List, Optional, Any
import structlog
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from core.llm_router import LLMRouter
from core.utils.json_cleaner import clean_json_string

logger = structlog.get_logger("deep-research")

_router_var: contextvars.ContextVar[Optional[LLMRouter]] = contextvars.ContextVar("diversify_router", default=None)
_router_loop_var: contextvars.ContextVar[Optional[asyncio.AbstractEventLoop]] = contextvars.ContextVar("diversify_router_loop", default=None)

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

_default_router = RouterProxy()

class QueryVariants(BaseModel):
    """Pydantic model representing a list of query variants."""
    variants: List[str] = Field(
        description="A list of 3 to 5 highly-targeted, distinct search query strings representing different facets or synonyms of the sub-question."
    )

async def diversify_query(
    question: str,
    num_variants: int = 3,
    router: Optional[Any] = None
) -> List[str]:
    """Diversify a single research sub-question into multiple distinct search query variants.
    
    Args:
        question: The research sub-question string.
        num_variants: The target number of variants to generate (clamped between 3 and 5).
        router: An optional LLMRouter or proxy (defaults to a global RouterProxy).
        
    Returns:
        List of distinct search query strings. On error, falls back to [question].
    """
    if not question or not question.strip():
        return []
        
    # Clamp num_variants between 3 and 5
    num_variants = max(3, min(5, num_variants))
    
    logger.info("Diversifying query", question=question, target_count=num_variants)
    
    active_router = router if router is not None else _default_router
    parser = JsonOutputParser(pydantic_object=QueryVariants)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert search engine optimizer and researcher. Your task is to diversify a single "
            "research sub-question into {num_variants} highly-targeted, distinct search queries.\n\n"
            "The queries should cover different aspects, synonyms, key technical terms, or perspectives of "
            "the question. They must be formatted for maximum search engine recall (e.g. keywords and short phrases, "
            "avoiding filler conversational words).\n\n"
            "CRITICAL: Do NOT start any search query with conversational or instructional verbs (e.g., 'explain', 'describe', 'define', 'discuss', 'analyze', 'summarize', 'find', 'search', 'get', 'what is', 'how to'). Standard search engines interpret these as instructions to find dictionary/glossary definitions, leading to low-quality, generic results. Instead, output direct nouns, technical terms, specifications, and factual keyword phrases.\n\n"
            "Format the output strictly as a JSON object matching the following schema:\n"
            "{format_instructions}"
        )),
        ("user", "Generate {num_variants} distinct search queries for this sub-question: '{question}'")
    ])
    
    formatted_prompt = prompt.format_prompt(
        question=question,
        num_variants=num_variants,
        format_instructions=parser.get_format_instructions()
    )
    
    try:
        response = await active_router.ainvoke(
            messages=formatted_prompt.to_messages(),
            tier="BULK",
            agent_name="SearchAgent",
            node_name="diversify_query"
        )
        
        parsed = parser.parse(clean_json_string(response.content))
        variants = parsed.get("variants", [])
        
        # Strip and filter empty variants
        clean_variants = [v.strip() for v in variants if v and v.strip()]
        
        if not clean_variants:
            logger.warning("No valid variants returned from LLM, falling back to original question")
            return [question]
            
        logger.info("Query diversification complete", original=question, variants=clean_variants)
        return clean_variants
        
    except Exception as e:
        logger.error("Failed to diversify query due to error, falling back to original question", error=str(e))
        return [question]
