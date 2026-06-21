import pytest
from unittest.mock import MagicMock, patch
from search.ddg import search_ddg, DDGSearchError
from core.models import SearchResult

@pytest.mark.asyncio
async def test_search_ddg_success():
    """Test successful DuckDuckGo search with mock results."""
    mock_results = [
        {"title": "Python Programming", "href": "https://python.org", "body": "Learn Python."},
        {"title": "Real Python", "href": "https://realpython.com", "body": "Python tutorials."}
    ]
    
    # Mock DDGS class and its text method
    with patch("search.ddg.DDGS") as mock_ddgs_cls:
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = mock_results
        mock_ddgs_cls.return_value.__enter__.return_value = mock_ddgs_instance
        
        results = await search_ddg("python programming", num_results=5)
        
        assert len(results) == 2
        
        # Verify first result
        assert results[0].title == "Python Programming"
        assert results[0].url == "https://python.org"
        assert results[0].content == "Learn Python."
        assert results[0].engine == "duckduckgo"
        
        # Verify second result
        assert results[1].title == "Real Python"
        assert results[1].url == "https://realpython.com"
        assert results[1].content == "Python tutorials."
        assert results[1].engine == "duckduckgo"
        
        # Verify text called correctly
        mock_ddgs_instance.text.assert_called_once_with("python programming", max_results=5)

@pytest.mark.asyncio
async def test_search_ddg_empty_query():
    """Test empty query handling on DuckDuckGo search."""
    with patch("search.ddg.DDGS") as mock_ddgs_cls:
        results = await search_ddg("   ")
        assert results == []
        mock_ddgs_cls.assert_not_called()

@pytest.mark.asyncio
async def test_search_ddg_error():
    """Test that exceptions raised during DuckDuckGo search are wrapped in DDGSearchError."""
    with patch("search.ddg.DDGS") as mock_ddgs_cls:
        mock_ddgs_cls.side_effect = RuntimeError("Service Unavailable")
        
        with pytest.raises(DDGSearchError, match="DuckDuckGo search failed"):
            await search_ddg("test query")
