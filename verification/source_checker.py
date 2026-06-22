import httpx
import structlog
from typing import Optional
from bs4 import BeautifulSoup

from core.models import VerifiedSource
from db.cache import SemanticCache

logger = structlog.get_logger("deep-research")

class SourceChecker:
    """Async source checker that verifies Cited URLs and retrieves their content, using a cache-first approach."""

    def __init__(self, cache: Optional[SemanticCache] = None) -> None:
        self.cache = cache or SemanticCache()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    async def check_source(self, url: str) -> VerifiedSource:
        """Verify URL accessibility and fetch content, checking cache first."""
        logger.info("Checking source URL", url=url)
        
        import os
        if os.environ.get("MOCK_LLM") == "true":
            logger.info("SourceChecker returning mock content in mock mode", url=url)
            mock_content = ""
            if "urllib.request" in url:
                mock_content = "This document shows that urllib.request with asyncio run_in_executor runs beautifully and asynchronously."
            elif "math" in url:
                mock_content = "This math reference proves that cosine similarity computed via math.sqrt and sum is extremely fast."
            else:
                mock_content = f"Mocked content for external source {url} with lightweight agent utilities."
            
            return VerifiedSource(
                url=url,
                title="Mocked Python Standard Library Documentation",
                content=mock_content,
                accessible=True,
                status_code=200
            )

        # 1. Cache-first lookup
        try:
            cached = await self.cache.get_url(url)
            if cached:
                logger.info("Source checker: Cache hit", url=url)
                return VerifiedSource(
                    url=url,
                    title=cached.get("title", "Cached Page"),
                    content=cached.get("content", ""),
                    accessible=True,
                    status_code=200
                )
        except Exception as cache_err:
            logger.warning("Cache lookup failed in SourceChecker", url=url, error=str(cache_err))

        # 2. Live HTTP GET fallback
        logger.info("Source checker: Cache miss, fetching live URL", url=url)
        async with httpx.AsyncClient(headers=self.headers, follow_redirects=True, timeout=5.0) as client:
            try:
                response = await client.get(url)
                if response.status_code >= 400:
                    logger.warn("Live URL returned error status", url=url, status_code=response.status_code)
                    return VerifiedSource(
                        url=url,
                        title="",
                        content="",
                        accessible=False,
                        status_code=response.status_code,
                        error_message=f"HTTP Error {response.status_code}"
                    )
                
                # Parse HTML content
                soup = BeautifulSoup(response.text, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else url
                
                # Extract text
                for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
                    script_or_style.decompose()
                content = soup.get_text(separator=" ").strip()
                # Clean up whitespace
                content = " ".join(content.split())

                # Cache newly fetched content
                try:
                    await self.cache.set_url(url, title, content)
                except Exception as cache_err:
                    logger.warning("Failed to write to cache in SourceChecker", url=url, error=str(cache_err))

                return VerifiedSource(
                    url=url,
                    title=title,
                    content=content,
                    accessible=True,
                    status_code=response.status_code
                )
                
            except httpx.HTTPError as http_err:
                logger.error("HTTP request failed in SourceChecker", url=url, error=str(http_err))
                return VerifiedSource(
                    url=url,
                    title="",
                    content="",
                    accessible=False,
                    status_code=None,
                    error_message=f"HTTP Error: {str(http_err)}"
                )
            except Exception as e:
                logger.error("Unexpected error in SourceChecker", url=url, error=str(e))
                return VerifiedSource(
                    url=url,
                    title="",
                    content="",
                    accessible=False,
                    status_code=None,
                    error_message=f"Unexpected error: {str(e)}"
                )
