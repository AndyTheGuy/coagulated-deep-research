import asyncio
from typing import Any, Dict, List, Tuple
import structlog
from langchain_core.messages import BaseMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from config.settings import settings, is_mock_llm_enabled

logger = structlog.get_logger("deep-research")

async def _invoke_with_retry(
    model: Any,
    messages: List[BaseMessage],
    retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    **kwargs: Any
) -> BaseMessage:
    """Invoke LLM with exponential backoff on exceptions."""
    delay = initial_delay
    for attempt in range(retries):
        try:
            return await model.ainvoke(messages, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            logger.warning(
                "LLM invocation failed, retrying...",
                attempt=attempt + 1,
                delay=delay,
                error=str(e)
            )
            await asyncio.sleep(delay)
            delay *= backoff_factor

def _extract_token_usage(response: Any) -> Tuple[int, int]:
    """Helper to extract input and output tokens from a LangChain response message."""
    # Try standard LangChain usage_metadata
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tok = response.usage_metadata.get("input_tokens", 0)
        output_tok = response.usage_metadata.get("output_tokens", 0)
        return input_tok, output_tok
        
    # Check response_metadata format
    metadata = getattr(response, "response_metadata", {})
    token_usage = metadata.get("token_usage", {})
    if token_usage:
        return token_usage.get("prompt_tokens", 0), token_usage.get("completion_tokens", 0)
        
    # Check alternative Vertex metadata format
    usage = metadata.get("usage_metadata", {})
    if usage:
        return usage.get("prompt_token_count", 0), usage.get("candidates_token_count", 0)
        
    return 0, 0

def resolve_vertex_model_name(model_name: str, tier: str) -> str:
    """Resolve and enforce native Gemini model names for Vertex AI, filtering out non-Gemini formats."""
    lower_name = model_name.lower()
    
    # Enforce mapping from generic placeholders or standard/bulk OpenAI defaults to native Gemini
    if lower_name in ("auto", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "claude-3-opus-20240229", "claude-3-sonnet-20240229"):
        if tier.upper() == "CRITICAL":
            return "gemini-3.5-flash"
        elif tier.upper() == "STANDARD":
            return "gemini-2.5-flash"
        else: # BULK
            return "gemini-2.5-flash-lite"
            
    # Map old/unapproved Gemini models to approved ones in the model garden
    if "gemini-1.5-flash" in lower_name:
        return "gemini-3.5-flash"
    if "gemini-1.5-pro" in lower_name:
        return "gemini-2.5-pro"
            
    # Extract native model ID if prefix or wrappers are used
    approved_models = (
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-image",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-image",
        "gemini-2.5-computer-use-preview-10-2025"
    )
    for native_id in approved_models:
        if native_id in lower_name:
            return native_id
            
    # Fallback default if completely unrecognizable but Vertex AI is requested
    return "gemini-3.5-flash"

class LLMRouter:
    """3-Tier router for LLM calls with automatic failover and token tracking."""
    
    def __init__(self) -> None:
        # Enforce native Gemini model names on Vertex AI
        critical_v = resolve_vertex_model_name(settings.CRITICAL_MODEL, "CRITICAL")
        standard_v = resolve_vertex_model_name(settings.STANDARD_MODEL, "STANDARD")
        bulk_v = resolve_vertex_model_name(settings.BULK_MODEL, "BULK")
        
        # Initialize LangChain Vertex model interfaces
        self._vertex_model = ChatVertexAI(
            model=critical_v,
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
        )
        
        self._vertex_standard = ChatVertexAI(
            model=standard_v,
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
        )
        
        self._vertex_bulk = ChatVertexAI(
            model=bulk_v,
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
        )
        
        # FreeLLMAPI configuration (only instantiated if enabled to save initialization checks)
        self._freellm_standard = None
        self._freellm_bulk = None
        if settings.USE_FREE_LLM_API:
            self._freellm_standard = ChatOpenAI(
                model=settings.STANDARD_MODEL,
                base_url=settings.FREE_LLM_API_BASE_URL,
                openai_api_key=settings.FREE_LLM_API_KEY,
                timeout=10.0,
            )
            self._freellm_bulk = ChatOpenAI(
                model=settings.BULK_MODEL,
                base_url=settings.FREE_LLM_API_BASE_URL,
                openai_api_key=settings.FREE_LLM_API_KEY,
                timeout=10.0,
            )
        
        # In-memory tracking dict for token usage metrics
        self.token_usage: Dict[str, Any] = {
            "vertex_ai": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
            "freellmapi": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
            "failovers": 0
        }

    async def ainvoke(
        self,
        messages: List[BaseMessage],
        tier: str = "STANDARD",
        agent_name: str = "unknown",
        node_name: str = "unknown",
        **kwargs: Any
    ) -> BaseMessage:
        """Route and execute the LLM invocation with automatic failover and native enforcement."""
        if is_mock_llm_enabled():
            response = self._get_mock_response(agent_name, node_name, messages)
            self._update_usage("freellmapi" if tier.upper() != "CRITICAL" else "vertex_ai", response)
            return response

        tier_upper = tier.upper()
        
        # Direct Vertex AI Mode (if FreeLLMAPI is completely disabled)
        if not settings.USE_FREE_LLM_API:
            if tier_upper == "CRITICAL":
                model_instance = self._vertex_model
            elif tier_upper == "STANDARD":
                model_instance = self._vertex_standard
            else: # BULK
                model_instance = self._vertex_bulk
                
            try:
                return await self._invoke_vertex(model_instance, messages, agent_name, node_name, tier=tier_upper, **kwargs)
            except Exception as e:
                # If standard or bulk fails on Vertex, fallback to the critical Vertex model as ultimate firewall
                if tier_upper != "CRITICAL":
                    logger.warn(
                        "Vertex AI call failed for standard/bulk tier, falling back to critical model",
                        tier=tier_upper,
                        agent=agent_name,
                        node=node_name,
                        error=str(e)
                    )
                    self.token_usage["failovers"] += 1
                    return await self._invoke_vertex(self._vertex_model, messages, agent_name, node_name, tier=tier_upper, **kwargs)
                raise e

        # Hybrid FreeLLMAPI Mode (if explicitly enabled)
        if tier_upper == "CRITICAL":
            try:
                return await self._invoke_vertex(self._vertex_model, messages, agent_name, node_name, tier=tier_upper, **kwargs)
            except Exception as e:
                logger.warn(
                    "Vertex AI direct call failed for CRITICAL tier, executing failover to FreeLLMAPI",
                    tier=tier_upper,
                    agent=agent_name,
                    node=node_name,
                    error=str(e)
                )
                self.token_usage["failovers"] += 1
                response = await _invoke_with_retry(self._freellm_standard, messages, retries=3, **kwargs)
                self._update_usage("freellmapi", response)
                return response
            
        model = self._freellm_standard if tier_upper == "STANDARD" else self._freellm_bulk
        try:
            logger.info(
                "Invoking FreeLLMAPI",
                tier=tier_upper,
                agent=agent_name,
                node=node_name,
                model=getattr(model, "model_name", "unknown")
            )
            response = await _invoke_with_retry(model, messages, retries=3, **kwargs)
            self._update_usage("freellmapi", response)
            return response
        except Exception as e:
            logger.warn(
                "FreeLLMAPI call failed, executing failover to Vertex AI",
                tier=tier_upper,
                agent=agent_name,
                node=node_name,
                error=str(e)
            )
            self.token_usage["failovers"] += 1
            # Fallback to corresponding Vertex model
            target_vertex_model = self._vertex_standard if tier_upper == "STANDARD" else self._vertex_bulk
            return await self._invoke_vertex(target_vertex_model, messages, agent_name, node_name, tier=tier_upper, **kwargs)

    async def _invoke_vertex(
        self,
        model_instance: ChatVertexAI,
        messages: List[BaseMessage],
        agent_name: str,
        node_name: str,
        tier: str,
        **kwargs: Any
    ) -> BaseMessage:
        """Helper to invoke a specific Google Vertex AI model directly."""
        logger.info(
            "Invoking Vertex AI",
            tier=tier,
            agent=agent_name,
            node=node_name,
            model=getattr(model_instance, "model", "unknown")
        )
        try:
            response = await _invoke_with_retry(model_instance, messages, retries=3, **kwargs)
            self._update_usage("vertex_ai", response)
            return response
        except Exception as e:
            logger.error(
                "Vertex AI invocation failed after all retries",
                tier=tier,
                agent=agent_name,
                node=node_name,
                error=str(e)
            )
            raise e

    def _update_usage(self, provider: str, response: BaseMessage) -> None:
        """Helper to parse and add token counts to statistics."""
        input_tok, output_tok = _extract_token_usage(response)
        self.token_usage[provider]["input_tokens"] += input_tok
        self.token_usage[provider]["output_tokens"] += output_tok
        self.token_usage[provider]["calls"] += 1
        
        logger.debug(
            "Tracked token usage",
            provider=provider,
            input_tokens=input_tok,
            output_tokens=output_tok
        )

    def _get_mock_response(self, agent_name: str, node_name: str, messages: Any) -> Any:
        from langchain_core.messages import AIMessage
        
        content = ""
        input_tokens = 50
        output_tokens = 50
        
        if agent_name == "ScopingAgent" and node_name == "clarify_with_user":
            content = '{"clarification_needed": false, "clarifying_question": null}'
            input_tokens, output_tokens = 15, 8
        elif agent_name == "ScopingAgent" and node_name == "write_research_brief":
            content = """{
                "topic": "Python Standard Library Architectures",
                "scope": "In-depth study on standard-library-first, zero-dependency, ultra-lightweight agentic Python systems.",
                "constraints": ["prioritize standard library solutions", "keep it simple"],
                "sub_questions": [
                    {"id": "q1", "question": "What standard modules are best for asynchronous HTTP requests?"},
                    {"id": "q2", "question": "How do we implement in-memory vector indexing with built-ins?"},
                    {"id": "q3", "question": "What is the optimal concurrency primitive for CPU-bound tasks?"}
                ],
                "target_source_count": 5
            }"""
            input_tokens, output_tokens = 45, 120
        elif agent_name == "SearchAgent" and node_name == "diversify_query":
            content = '{"variants": ["python asyncio http", "python lightweight agent vector", "python multiprocessing vs threading concurrency"]}'
            input_tokens, output_tokens = 25, 20
        elif agent_name == "ResearcherAgent" and node_name == "summarize_sources":
            content = "For async HTTP, urllib.request with asyncio run_in_executor or standard socket-level connections are lightweight. For indexing, flat list embeddings with cosine similarity computed via math.sqrt and sum are simple and effective. Multiprocessing is best for CPU-bound tasks due to the GIL, while asyncio is optimal for I/O bound tasks."
            input_tokens, output_tokens = 120, 80
        elif agent_name == "VerifierAgent" and node_name == "claim_extraction":
            content = """{
                "claims": [
                    {
                        "claim_id": "c1",
                        "claim_text": "urllib.request runs synchronously but can be run concurrently via asyncio run_in_executor.",
                        "section": "Asynchronous HTTP",
                        "supporting_quotes": ["urllib.request with asyncio run_in_executor"],
                        "source_urls": ["https://docs.python.org/3/library/urllib.request.html"]
                    },
                    {
                        "claim_id": "c2",
                        "claim_text": "Cosine similarity can be implemented natively with math.sqrt.",
                        "section": "In-memory Indexing",
                        "supporting_quotes": ["cosine similarity computed via math.sqrt and sum"],
                        "source_urls": ["https://docs.python.org/3/library/math.html"]
                    }
                ]
            }"""
            input_tokens, output_tokens = 150, 140
        elif agent_name == "VerifierAgent" and node_name == "verifier_critique":
            content = '{"gaps_found": false, "critique_text": "Verified successfully.", "suggested_queries": []}'
            input_tokens, output_tokens = 180, 15
        elif agent_name == "WriterAgent" and node_name == "report_writer":
            content = """{
                "title": "Autonomous Python Standard Library Agent Architectures",
                "content": "# Autonomous Python Standard Library Agent Architectures\\n\\n## 1. Asynchronous HTTP Requests\\nFor async HTTP requests, although packages like `aiohttp` or `httpx` are common, the standard library offers `urllib.request` which can be executed concurrently in a thread pool via `asyncio.get_running_loop().run_in_executor()` [1]. This maintains zero external dependencies.\\n\\n## 2. Flat Vector Indexing\\nVector operations can be implemented natively. A simple dot product and cosine similarity function using `math.sqrt` [2] allows full-text flat vector searching in less than 10 lines of Python. This avoids importing dense frameworks like `numpy` or `scipy` for lightweight agent systems.\\n\\n## 3. Concurrency GIL Barriers\\nTo scale CPU-intensive tasks, agents must spawn sub-processes via `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`, bypassing the Global Interpreter Lock (GIL)."
            }"""
            input_tokens, output_tokens = 250, 300
        elif agent_name == "EvaluatorAgent":
            if node_name == "extract_key_facts":
                content = '{"key_facts": ["urllib.request is standard library", "cosine similarity computed via math.sqrt", "multiprocessing bypasses GIL"]}'
                input_tokens, output_tokens = 80, 40
            elif node_name == "check_coverage":
                content = """{
                    "coverage_items": [
                        {"fact": "urllib.request is standard library", "covered": true, "explanation": "The report covers urllib.request under async HTTP."},
                        {"fact": "cosine similarity computed via math.sqrt", "covered": true, "explanation": "The report covers native cosine similarity using math.sqrt."},
                        {"fact": "multiprocessing bypasses GIL", "covered": true, "explanation": "The report mentions multiprocessing ProcessPoolExecutor to bypass the GIL."}
                    ]
                }"""
                input_tokens, output_tokens = 150, 100
            elif node_name == "evaluate_reasoning":
                content = '{"score": 0.95, "explanation": "The report has outstanding coherence and is logically sound based on verified claims."}'
                input_tokens, output_tokens = 120, 30
            elif node_name == "evaluate_factuality":
                content = '{"score": 0.98, "explanation": "Inline citations are perfectly mapped to standard bibliography items."}'
                input_tokens, output_tokens = 120, 30
            elif node_name == "remediation_formulation":
                content = '{"gaps_found": false, "remediation_queries": [], "remediation_notes": "Passed all metrics!"}'
                input_tokens, output_tokens = 150, 20
            else:
                content = '{"status": "success"}'
                input_tokens, output_tokens = 10, 5
        elif agent_name == "MCTSPlanner" and node_name == "generate_candidate_intents":
            content = "Determine best lightweight modules for async HTTP requests\nAnalyze native vector indexing using math functions\nEvaluate standard multiprocessing vs multithreading for CPU bound work"
            input_tokens, output_tokens = 30, 45
        elif agent_name == "MCTSPlanner" and node_name == "simulate_outcome_state":
            content = "Simulated progress: identified urllib.request and ProcessPoolExecutor as built-in standard library solutions. Satisfied."
            input_tokens, output_tokens = 35, 25
        elif agent_name == "MCTSPlanner" and node_name == "evaluate_state_quality":
            content = "0.85"
            input_tokens, output_tokens = 20, 5
        elif agent_name == "MCTSPlanner" and node_name == "generate_queries_for_intent":
            content = "python urllib request async\npython in-memory vector index math\npython multiprocessing vs multithreading"
            input_tokens, output_tokens = 30, 25
        elif agent_name == "MCTSPlanner" and node_name == "repair_plan":
            content = "lightweight python async http\nnative python vector similarity\npython concurrency GIL subprocess"
            input_tokens, output_tokens = 30, 25
        else:
            content = '{"status": "success"}'
            input_tokens, output_tokens = 10, 5
            
        return AIMessage(
            content=content,
            usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
        )
