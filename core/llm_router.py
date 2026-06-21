import asyncio
from typing import Any, Dict, List, Tuple
import structlog
from langchain_core.messages import BaseMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from config.settings import settings

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

class LLMRouter:
    """3-Tier router for LLM calls with automatic failover and token tracking."""
    
    def __init__(self) -> None:
        # Initialize LangChain model interfaces
        self._vertex_model = ChatVertexAI(
            model=settings.CRITICAL_MODEL,
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
        )
        
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
        """Route and execute the LLM invocation with failover for non-critical tiers."""
        import os
        if os.environ.get("MOCK_LLM") == "true":
            response = self._get_mock_response(agent_name, node_name, messages)
            self._update_usage("freellmapi" if tier.upper() != "CRITICAL" else "vertex_ai", response)
            return response

        tier_upper = tier.upper()
        
        # Tier 1: CRITICAL goes strictly to Google Vertex AI with failover to FreeLLMAPI if Vertex fails
        if tier_upper == "CRITICAL":
            try:
                return await self._invoke_vertex(messages, agent_name, node_name, tier=tier_upper, **kwargs)
            except Exception as e:
                logger.warn(
                    "Vertex AI direct call failed for CRITICAL tier, executing failover to FreeLLMAPI",
                    tier=tier_upper,
                    agent=agent_name,
                    node=node_name,
                    error=str(e)
                )
                self.token_usage["failovers"] += 1
                # Fallback to FreeLLMAPI standard
                model = self._freellm_standard
                response = await _invoke_with_retry(model, messages, retries=3, **kwargs)
                self._update_usage("freellmapi", response)
                return response
            
        # Tiers 2 & 3: STANDARD and BULK try FreeLLMAPI first
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
            # Fallback to Vertex AI
            return await self._invoke_vertex(messages, agent_name, node_name, tier=tier_upper, **kwargs)

    async def _invoke_vertex(
        self,
        messages: List[BaseMessage],
        agent_name: str,
        node_name: str,
        tier: str = "CRITICAL",
        **kwargs: Any
    ) -> BaseMessage:
        """Helper to invoke Google Vertex AI directly."""
        logger.info(
            "Invoking Vertex AI",
            tier=tier,
            agent=agent_name,
            node=node_name,
            model=getattr(self._vertex_model, "model", "unknown")
        )
        try:
            response = await _invoke_with_retry(self._vertex_model, messages, retries=3, **kwargs)
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
            content = """{
                "key_information_coverage": {
                    "score": 0.92,
                    "threshold": 0.80,
                    "passed": true
                },
                "reasoning_quality": {
                    "score": 0.88,
                    "threshold": 0.75,
                    "passed": true
                },
                "factuality": {
                    "score": 0.95,
                    "threshold": 0.90,
                    "passed": true
                },
                "overall_passed": true,
                "evaluator_notes": "Highly precise report aligning perfectly with modern standard-library constraints. Accurate citation mappings."
            }"""
            input_tokens, output_tokens = 400, 180
        else:
            content = '{"status": "success"}'
            input_tokens, output_tokens = 10, 5
            
        return AIMessage(
            content=content,
            usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
        )
