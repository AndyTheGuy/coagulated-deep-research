from typing import Any, Dict, List, Tuple
import structlog
from langchain_core.messages import BaseMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from config.settings import settings

logger = structlog.get_logger("deep-research")

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
            openai_api_base=settings.FREE_LLM_API_BASE_URL,
            openai_api_key=settings.FREE_LLM_API_KEY,
            timeout=10.0,
        )
        
        self._freellm_bulk = ChatOpenAI(
            model=settings.BULK_MODEL,
            openai_api_base=settings.FREE_LLM_API_BASE_URL,
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
        tier_upper = tier.upper()
        
        # Tier 1: CRITICAL goes strictly to Google Vertex AI
        if tier_upper == "CRITICAL":
            return await self._invoke_vertex(messages, agent_name, node_name, **kwargs)
            
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
            response = await model.ainvoke(messages, **kwargs)
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
            return await self._invoke_vertex(messages, agent_name, node_name, **kwargs)

    async def _invoke_vertex(
        self,
        messages: List[BaseMessage],
        agent_name: str,
        node_name: str,
        **kwargs: Any
    ) -> BaseMessage:
        """Helper to invoke Google Vertex AI directly."""
        logger.info(
            "Invoking Vertex AI",
            tier="CRITICAL",
            agent=agent_name,
            node=node_name,
            model=getattr(self._vertex_model, "model", "unknown")
        )
        response = await self._vertex_model.ainvoke(messages, **kwargs)
        self._update_usage("vertex_ai", response)
        return response

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
