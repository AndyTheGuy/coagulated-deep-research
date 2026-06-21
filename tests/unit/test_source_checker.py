import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from core.models import VerifiedSource
from verification.source_checker import SourceChecker

@pytest.mark.asyncio
async def test_source_checker_cache_hit():
    # Arrange
    mock_cache = MagicMock()
    mock_cache.get_url = AsyncMock(return_value={
        "title": "Test Title",
        "content": "This is test cached content of the page."
    })
    
    checker = SourceChecker(cache=mock_cache)
    url = "https://example.com/cached"
    
    # Act
    result = await checker.check_source(url)
    
    # Assert
    assert isinstance(result, VerifiedSource)
    assert result.url == url
    assert result.title == "Test Title"
    assert result.content == "This is test cached content of the page."
    assert result.accessible is True
    assert result.status_code == 200
    
    mock_cache.get_url.assert_called_once_with(url)
    mock_cache.set_url.assert_not_called()

@pytest.mark.asyncio
async def test_source_checker_cache_miss_live_success():
    # Arrange
    mock_cache = MagicMock()
    mock_cache.get_url = AsyncMock(return_value=None)
    mock_cache.set_url = AsyncMock()
    
    checker = SourceChecker(cache=mock_cache)
    url = "https://example.com/live"
    
    html_content = "<html><head><title>Live Title</title></head><body><script>javascript</script>Main content to scrape</body></html>"
    
    # Mock httpx AsyncClient
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = html_content
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    
    # Patch client context manager
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_ctx = AsyncMock()
        mock_client_ctx.__aenter__.return_value = mock_client
        mock_client_class.return_value = mock_client_ctx
        
        # Act
        result = await checker.check_source(url)
        
        # Assert
        assert isinstance(result, VerifiedSource)
        assert result.url == url
        assert result.title == "Live Title"
        assert result.content == "Live Title Main content to scrape"
        assert result.accessible is True
        assert result.status_code == 200
        
        mock_cache.get_url.assert_called_once_with(url)
        mock_cache.set_url.assert_called_once_with(url, "Live Title", "Live Title Main content to scrape")

@pytest.mark.asyncio
async def test_source_checker_cache_miss_live_http_error():
    # Arrange
    mock_cache = MagicMock()
    mock_cache.get_url = AsyncMock(return_value=None)
    mock_cache.set_url = AsyncMock()
    
    checker = SourceChecker(cache=mock_cache)
    url = "https://example.com/404"
    
    mock_response = MagicMock()
    mock_response.status_code = 404
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_ctx = AsyncMock()
        mock_client_ctx.__aenter__.return_value = mock_client
        mock_client_class.return_value = mock_client_ctx
        
        # Act
        result = await checker.check_source(url)
        
        # Assert
        assert isinstance(result, VerifiedSource)
        assert result.url == url
        assert result.accessible is False
        assert result.status_code == 404
        assert "HTTP Error 404" in result.error_message
        
        mock_cache.get_url.assert_called_once_with(url)
        mock_cache.set_url.assert_not_called()

@pytest.mark.asyncio
async def test_source_checker_live_exception():
    # Arrange
    mock_cache = MagicMock()
    mock_cache.get_url = AsyncMock(return_value=None)
    
    checker = SourceChecker(cache=mock_cache)
    url = "https://example.com/timeout"
    
    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectTimeout("Timeout connecting"))
    
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client_ctx = AsyncMock()
        mock_client_ctx.__aenter__.return_value = mock_client
        mock_client_class.return_value = mock_client_ctx
        
        # Act
        result = await checker.check_source(url)
        
        # Assert
        assert isinstance(result, VerifiedSource)
        assert result.url == url
        assert result.accessible is False
        assert result.status_code is None
        assert "HTTP Error" in result.error_message
