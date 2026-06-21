import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from search.scraper import scrape_url, ScrapingError, html_to_clean_markdown

def test_html_to_clean_markdown():
    """Test HTML cleaning and conversion to structured markdown."""
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quantum Computing Intro</title>
        <style>body { font-family: sans-serif; }</style>
    </head>
    <body>
        <header>
            <nav><a href="/home">Home</a></nav>
        </header>
        <main>
            <h1>Understanding Quantum</h1>
            <p>Quantum computing utilizes quantum mechanics. Here is a <a href="https://example.com/learn">tutorial link</a>.</p>
            <h2>Core Concepts</h2>
            <ul>
                <li>Superposition</li>
                <li>Entanglement</li>
            </ul>
        </main>
        <footer>
            <p>&copy; 2026 Quantum Inc.</p>
        </footer>
        <script>console.log("hello");</script>
    </body>
    </html>
    """
    
    title, markdown = html_to_clean_markdown(sample_html)
    
    assert title == "Quantum Computing Intro"
    
    # Check that headings, paragraphs, lists, and links are formatted, while style/script/nav/footer/header are stripped
    assert "# Understanding Quantum" in markdown
    assert "## Core Concepts" in markdown
    assert "Quantum computing utilizes quantum mechanics." in markdown
    assert "[tutorial link](https://example.com/learn)" in markdown
    assert "* Superposition" in markdown
    assert "* Entanglement" in markdown
    
    # Assert stripped tags are not present
    assert "body {" not in markdown
    assert "console.log" not in markdown
    assert "Quantum Inc" not in markdown
    assert "Home" not in markdown

@pytest.mark.asyncio
async def test_scrape_url_success():
    """Test successful URL scraping."""
    sample_html = "<html><head><title>Test Title</title></head><body><p>Hello World</p></body></html>"
    
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.text = sample_html
    mock_response.raise_for_status = MagicMock()
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
        
        title, content = await scrape_url("https://example.com")
        
        assert title == "Test Title"
        assert content == "Hello World"
        
        # Verify headers (User-Agent spoofing)
        called_args, called_kwargs = mock_get.call_args
        assert "User-Agent" in called_kwargs["headers"]

@pytest.mark.asyncio
async def test_scrape_url_http_error():
    """Test HTTPStatusError mapping to ScrapingError."""
    mock_request = httpx.Request("GET", "https://example.com")
    mock_response = httpx.Response(404, request=mock_request)
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.HTTPStatusError(
            message="Not Found",
            request=mock_request,
            response=mock_response
        )
        
        with pytest.raises(ScrapingError, match="HTTP status 404"):
            await scrape_url("https://example.com/missing")

@pytest.mark.asyncio
async def test_scrape_url_network_error():
    """Test network timeout mapping to ScrapingError."""
    mock_request = httpx.Request("GET", "https://example.com")
    
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectTimeout("Connect timeout", request=mock_request)
        
        with pytest.raises(ScrapingError, match="network request failed"):
            await scrape_url("https://example.com")
