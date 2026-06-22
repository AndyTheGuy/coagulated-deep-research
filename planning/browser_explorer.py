import asyncio
import structlog
from typing import Any, Dict, List, Optional
from core.mcp_client import MCPHub
from search.scraper import scrape_url
from config.settings import is_mock_llm_enabled

logger = structlog.get_logger("deep-research")

class BrowserExplorer:
    """Bridges the high-level Plan-MCTS planning engine with the headless browser MCP server.
    Provides fallback mechanism to standard web scraper if Puppeteer is not available.
    """
    
    def __init__(self, mcp_hub: Optional[MCPHub] = None, scrape_func=None):
        self.mcp_hub = mcp_hub or MCPHub()
        self.scrape_func = scrape_func

    async def explore_url(self, url: str) -> Dict[str, Any]:
        """Navigates to a URL, extracts page HTML or text, and logs progress.
        Uses Puppeteer MCP if available; otherwise falls back to basic scrapers.
        """
        logger.info("BrowserExplorer exploring URL", url=url)
        
        if is_mock_llm_enabled():
            logger.info("BrowserExplorer returning mock content in mock mode", url=url)
            mock_content = ""
            if "urllib.request" in url:
                mock_content = "This document shows that urllib.request with asyncio run_in_executor runs beautifully and asynchronously."
            elif "math" in url:
                mock_content = "This math reference proves that cosine similarity computed via math.sqrt and sum is extremely fast."
            else:
                mock_content = f"Mocked content for external source {url} with lightweight agent utilities."
            
            return {
                "url": url,
                "title": "Mocked Python Standard Library Documentation",
                "content": mock_content,
                "method": "mock_mode",
                "success": True
            }

        try:
            # Try Puppeteer MCP server first
            puppeteer_client = await self.mcp_hub.get_client("puppeteer")
            if getattr(puppeteer_client, "mock", False) is True:
                logger.info("Puppeteer MCP client is in mock mode; forcing graceful fallback to standard scraper")
                raise RuntimeError("Puppeteer MCP client is mocked")
                
            logger.debug("Puppeteer MCP server available, executing navigation")
            
            # 1. Navigate
            nav_res = await puppeteer_client.call_tool("navigate", {"url": url})
            logger.debug("Puppeteer navigation complete", result=nav_res)
            
            # 2. Extract content (using get_html or similar)
            html_res = await puppeteer_client.call_tool("get_html", {})
            html_content = html_res.get("content", [{}])[0].get("text", "")
            
            # Simple title extraction from raw HTML if needed
            title = "Extracted Page"
            if "<title>" in html_content.lower():
                try:
                    title = html_content.split("<title>")[1].split("</title>")[0].strip()
                except Exception:
                    pass
                    
            return {
                "url": url,
                "title": title,
                "content": html_content,
                "method": "puppeteer",
                "success": True
            }
            
        except Exception as e:
            logger.warn("Puppeteer exploration failed or unavailable, falling back to basic scraper", url=url, error=str(e))
            
            # Graceful fallback to native beautifulsoup/httpx scrape
            try:
                scrape_fn = self.scrape_func or scrape_url
                title, text = await scrape_fn(url)
                if not text or not text.strip():
                    raise ValueError("Scraped content is empty")
                    
                return {
                    "url": url,
                    "title": title or "Unknown Title",
                    "content": text,
                    "method": "scraper_fallback",
                    "success": True
                }
            except Exception as ex:
                logger.error("All scraper pathways failed for URL", url=url, error=str(ex))
                return {
                    "url": url,
                    "title": "Unreachable Page",
                    "content": "",
                    "method": "failed",
                    "success": False,
                    "error_message": str(ex)
                }

    async def click_element(self, selector: str) -> bool:
        """Clicks an element in the browser. Only supported under Puppeteer."""
        try:
            puppeteer_client = await self.mcp_hub.get_client("puppeteer")
            if getattr(puppeteer_client, "mock", False) is True:
                logger.info("Click operation bypassed because Puppeteer MCP client is mocked")
                return False
            await puppeteer_client.call_tool("click", {"selector": selector})
            logger.info("BrowserExplorer successfully clicked selector", selector=selector)
            return True
        except Exception as e:
            logger.error("Click operation failed or not supported in current mode", selector=selector, error=str(e))
            return False
