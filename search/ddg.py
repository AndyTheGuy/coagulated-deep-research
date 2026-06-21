import asyncio
from typing import List, Optional
import structlog
from duckduckgo_search import DDGS
from core.models import SearchResult
from search.searxng import SearchError

logger = structlog.get_logger("deep-research")

class DDGSearchError(SearchError):
    """Exception raised for errors in the DuckDuckGo client."""
    pass

def _sync_ddg_search(query: str, num_results: int) -> List[dict]:
    """Execute synchronous DuckDuckGo search."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=num_results)
            return list(results) if results else []
    except Exception as e:
        raise DDGSearchError(f"DuckDuckGo search failed: {e}") from e

async def search_ddg(query: str, num_results: int = 10) -> List[SearchResult]:
    """Search DuckDuckGo asynchronously and return standardized SearchResult list.
    
    Args:
        query: The search query string.
        num_results: Maximum number of results to return.
        
    Returns:
        List[SearchResult]
        
    Raises:
        DDGSearchError: If DuckDuckGo search fails.
    """
    if not query or not query.strip():
        return []
        
    logger.info("Querying DuckDuckGo (fallback)", query=query, num_results=num_results)
    
    try:
        raw_results = await asyncio.to_thread(_sync_ddg_search, query, num_results)
        
        search_results = []
        for item in raw_results:
            title = item.get("title", "")
            url = item.get("href", "")
            content = item.get("body", "")
            
            search_results.append(
                SearchResult(
                    title=title,
                    url=url,
                    content=content,
                    score=None,
                    engine="duckduckgo"
                )
            )
            
        logger.info("DuckDuckGo search successful", count=len(search_results))
        return search_results
        
    except Exception as e:
        logger.error("DuckDuckGo search failed", error=str(e))
        if not isinstance(e, DDGSearchError):
            raise DDGSearchError(f"Unexpected DuckDuckGo error: {e}") from e
        raise
