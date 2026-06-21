import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from search.searxng import search_searxng, SearXNGError
from core.models import SearchResult

@pytest.mark.asyncio
async def test_search_searxng_success():
    """Test successful SearXNG search with valid results."""
    mock_response_data = {
        "results": [
            {
                "title": "Quantum Computing",
                "url": "https://example.com/quantum",
                "content": "An introduction to quantum computing.",
                "score": 0.95,
                "engines": ["google", "wikipedia"]
            },
            {
                "title": "Quantum Supremacy",
                "url": "https://example.com/supremacy",
                "snippet": "Achieving quantum supremacy.",
                "score": 0.88,
                "engine": "wikipedia"
            }
        ]
    }
    
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = MagicMock(return_value=mock_response_data)
    mock_response.raise_for_status = MagicMock()
    
    # Mock httpx.AsyncClient.get
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        
        results = await search_searxng("quantum physics", engines=["google", "wikipedia"], num_results=10)
        
        assert len(results) == 2
        
        # Verify first result mapping
        assert results[0].title == "Quantum Computing"
        assert results[0].url == "https://example.com/quantum"
        assert results[0].content == "An introduction to quantum computing."
        assert results[0].score == 0.95
        assert results[0].engine == "google"  # Derived from engines list
        
        # Verify second result mapping (with snippet fallback and explicit engine)
        assert results[1].title == "Quantum Supremacy"
        assert results[1].url == "https://example.com/supremacy"
        assert results[1].content == "Achieving quantum supremacy."
        assert results[1].score == 0.88
        assert results[1].engine == "wikipedia"
        
        # Verify mock_get was called with correct parameters
        mock_get.assert_called_once()
        called_args, called_kwargs = mock_get.call_args
        assert called_kwargs["params"]["q"] == "quantum physics"
        assert called_kwargs["params"]["format"] == "json"
        assert called_kwargs["params"]["engines"] == "google,wikipedia"

@pytest.mark.asyncio
async def test_search_searxng_empty_query():
    """Test that empty queries return an empty list immediately without network request."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        results = await search_searxng("   ")
        assert results == []
        mock_get.assert_not_called()

@pytest.mark.asyncio
async def test_search_searxng_http_error():
    """Test that HTTPStatusError is caught and raised as SearXNGError."""
    # Create mock response and request for HTTPStatusError
    mock_request = httpx.Request("GET", "http://localhost:8080")
    mock_response = httpx.Response(500, request=mock_request)
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.HTTPStatusError(
            message="Internal Server Error",
            request=mock_request,
            response=mock_response
        )
        
        with pytest.raises(SearXNGError, match="HTTP 500"):
            await search_searxng("test")

@pytest.mark.asyncio
async def test_search_searxng_network_error():
    """Test that RequestError is caught and raised as SearXNGError."""
    mock_request = httpx.Request("GET", "http://localhost:8080")
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("Connection timed out", request=mock_request)
        
        with pytest.raises(SearXNGError, match="network request failed"):
            await search_searxng("test")

@pytest.mark.asyncio
async def test_search_searxng_invalid_json():
    """Test that JSON decode error is caught and raised as SearXNGError."""
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json = MagicMock(side_effect=ValueError("Invalid JSON"))
    mock_response.raise_for_status = MagicMock()
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        
        with pytest.raises(SearXNGError, match="Failed to parse SearXNG JSON"):
            await search_searxng("test")
