import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from planning.browser_explorer import BrowserExplorer

@pytest.mark.asyncio
async def test_browser_explorer_puppeteer_success():
    """Test successful URL exploration using the Puppeteer MCP client."""
    mock_hub = MagicMock()
    mock_client = MagicMock()
    
    # Mock programmatic async method call_tool
    mock_client.call_tool = AsyncMock()
    
    # First call is navigate, second call is get_html
    mock_client.call_tool.side_effect = [
        {"success": True, "content": []}, # navigate result
        {"success": True, "content": [{"text": "<html><head><title>Mock Page Title</title></head><body>Hello World</body></html>"}]} # get_html result
    ]
    
    mock_hub.get_client = AsyncMock(return_value=mock_client)
    
    explorer = BrowserExplorer(mcp_hub=mock_hub)
    res = await explorer.explore_url("https://example.com/test")
    
    assert res["success"] is True
    assert res["url"] == "https://example.com/test"
    assert res["title"] == "Mock Page Title"
    assert "Hello World" in res["content"]
    assert res["method"] == "puppeteer"
    
    mock_hub.get_client.assert_called_once_with("puppeteer")
    assert mock_client.call_tool.call_count == 2

@pytest.mark.asyncio
async def test_browser_explorer_fallback_success():
    """Test graceful fallback to standard scrape_url when Puppeteer client fails."""
    mock_hub = MagicMock()
    mock_hub.get_client = AsyncMock(side_effect=Exception("Puppeteer server not available"))
    
    explorer = BrowserExplorer(mcp_hub=mock_hub)
    
    # Mock standard scrape_url
    with patch("planning.browser_explorer.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = ("Fallback Page Title", "Fallback scraped content text.")
        
        res = await explorer.explore_url("https://example.com/fallback")
        
        assert res["success"] is True
        assert res["url"] == "https://example.com/fallback"
        assert res["title"] == "Fallback Page Title"
        assert res["content"] == "Fallback scraped content text."
        assert res["method"] == "scraper_fallback"
        
        mock_scrape.assert_called_once_with("https://example.com/fallback")

@pytest.mark.asyncio
async def test_browser_explorer_all_failed():
    """Test response when both Puppeteer and fallback scraper fail."""
    mock_hub = MagicMock()
    mock_hub.get_client = AsyncMock(side_effect=Exception("Puppeteer server down"))
    
    explorer = BrowserExplorer(mcp_hub=mock_hub)
    
    with patch("planning.browser_explorer.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.side_effect = Exception("HTTP Connection Timeout")
        
        res = await explorer.explore_url("https://example.com/fail")
        
        assert res["success"] is False
        assert res["url"] == "https://example.com/fail"
        assert res["title"] == "Unreachable Page"
        assert res["content"] == ""
        assert res["method"] == "failed"
        assert "HTTP Connection Timeout" in res["error_message"]

@pytest.mark.asyncio
async def test_browser_explorer_click_element_success():
    """Test clicking an element via Puppeteer."""
    mock_hub = MagicMock()
    mock_client = MagicMock()
    mock_client.call_tool = AsyncMock(return_value={"success": True})
    mock_hub.get_client = AsyncMock(return_value=mock_client)
    
    explorer = BrowserExplorer(mcp_hub=mock_hub)
    success = await explorer.click_element("#submit-button")
    
    assert success is True
    mock_client.call_tool.assert_called_once_with("click", {"selector": "#submit-button"})

@pytest.mark.asyncio
async def test_browser_explorer_click_element_fail():
    """Test clicking fails gracefully if Puppeteer raises an exception."""
    mock_hub = MagicMock()
    mock_hub.get_client = AsyncMock(side_effect=Exception("Element not found"))
    
    explorer = BrowserExplorer(mcp_hub=mock_hub)
    success = await explorer.click_element("#invalid-btn")
    
    assert success is False
