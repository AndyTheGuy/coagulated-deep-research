import httpx
from typing import List, Optional
import structlog
from config.settings import settings
from core.models import SearchResult

logger = structlog.get_logger("deep-research")

class SearchError(Exception):
    """Base exception for search operations."""
    pass

class SearXNGError(SearchError):
    """Exception raised for errors in the SearXNG client."""
    pass

async def search_searxng(
    query: str, 
    engines: Optional[List[str]] = None, 
    num_results: int = 10
) -> List[SearchResult]:
    """Search SearXNG asynchronously and return standardized SearchResult list.
    
    Args:
        query: The search query string.
        engines: Optional list of engine names to query.
        num_results: Maximum number of results to return.
        
    Returns:
        List[SearchResult]
        
    Raises:
        SearXNGError: If network request fails, times out, or response format is invalid.
    """
    if not query or not query.strip():
        return []
        
    url = f"{settings.SEARXNG_URL.rstrip('/')}/search"
    params = {
        "q": query,
        "format": "json"
    }
    if engines:
        params["engines"] = ",".join(engines)
        
    logger.info("Querying SearXNG", query=query, url=url, engines=engines)
    
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            try:
                data = response.json()
            except ValueError as e:
                logger.error("SearXNG response parsing failure", error=str(e))
                raise SearXNGError(f"Failed to parse SearXNG JSON response: {e}") from e
                
            results_data = data.get("results", [])
            
            search_results = []
            for item in results_data[:num_results]:
                title = item.get("title", "")
                item_url = item.get("url", "")
                # Fallback to snippet if content is missing
                content = item.get("content", item.get("snippet", ""))
                score = item.get("score")
                
                # Extract engine: first engine from engines list, or engine field
                item_engines = item.get("engines", [])
                engine = item.get("engine")
                if not engine and item_engines:
                    engine = item_engines[0]
                    
                search_results.append(
                    SearchResult(
                        title=title,
                        url=item_url,
                        content=content,
                        score=score,
                        engine=engine
                    )
                )
                
            logger.info("SearXNG search successful", count=len(search_results))
            return search_results
            
    except httpx.HTTPStatusError as e:
        logger.error("SearXNG server error", status_code=e.response.status_code, error=str(e))
        raise SearXNGError(f"SearXNG server error (HTTP {e.response.status_code}): {e}") from e
    except httpx.RequestError as e:
        logger.error("SearXNG network request failure", error=str(e))
        raise SearXNGError(f"SearXNG network request failed: {e}") from e
    except SearXNGError:
        raise
    except Exception as e:
        logger.error("Unexpected error in SearXNG client", error=str(e))
        raise SearXNGError(f"Unexpected SearXNG client error: {e}") from e
