import re
import httpx
from bs4 import BeautifulSoup, Comment
from typing import Optional, Tuple
import structlog

logger = structlog.get_logger("deep-research")

class ScrapingError(Exception):
    """Base exception for scraping operations."""
    pass

def html_to_clean_markdown(html_content: str) -> Tuple[str, str]:
    """Convert HTML content to clean, readable plain text/markdown.
    
    Args:
        html_content: Raw HTML page source.
        
    Returns:
        Tuple of (page_title, cleaned_markdown_text)
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text().strip() if title_tag else ""
    
    # Remove unwanted tags
    unwanted_tags = [
        "script", "style", "nav", "footer", "header", "aside", 
        "noscript", "svg", "iframe", "form", "button"
    ]
    for tag in soup(unwanted_tags):
        tag.decompose()
        
    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
        
    # Standardize and build clean markdown-like structure
    lines = []
    
    # Traverse element by element
    body = soup.find("body") or soup
    
    for element in body.descendants:
        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # Check if this element was already processed by a parent container
            text = element.get_text().strip()
            if text:
                level = int(element.name[1])
                lines.append(f"\n\n{'#' * level} {text}\n\n")
        elif element.name == "p":
            text = element.get_text().strip()
            if text:
                lines.append(f"\n\n{text}\n\n")
        elif element.name == "li":
            text = element.get_text().strip()
            if text:
                lines.append(f"\n* {text}\n")
        elif element.name == "a":
            # Avoid processing nested a elements redundantly
            href = element.get("href", "")
            text = element.get_text().strip()
            if text and href and href.startswith("http"):
                lines.append(f" [{text}]({href}) ")
        elif element.name is None:  # Text node
            # Only add raw text if its parent is not one of the custom handled block tags
            parent = element.parent
            if parent and parent.name not in [
                "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "a", 
                "script", "style", "html", "head", "title"
            ]:
                text = element.strip()
                if text:
                    lines.append(f" {text} ")
                    
    # Join and clean up whitespace
    text_content = "".join(lines)
    
    # Collapse multiple spaces
    text_content = re.sub(r"[ \t]+", " ", text_content)
    # Collapse multiple consecutive newlines (limit to max 2)
    text_content = re.sub(r"\n{3,}", "\n\n", text_content)
    
    return title, text_content.strip()

async def scrape_url(url: str, timeout: float = 10.0) -> Tuple[str, str]:
    """Scrape a URL asynchronously and extract clean text/markdown content.
    
    Args:
        url: The URL to fetch and scrape.
        timeout: Network timeout in seconds.
        
    Returns:
        Tuple of (title, clean_markdown_content)
        
    Raises:
        ScrapingError: If fetching or scraping fails.
    """
    logger.info("Scraping URL", url=url)
    
    # Spoof User-Agent to avoid blocks
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            html_content = response.text
            title, content = html_to_clean_markdown(html_content)
            
            logger.info("Scraping successful", url=url, title=title, content_len=len(content))
            return title, content
            
    except httpx.HTTPStatusError as e:
        logger.error("Scraper received error HTTP status", url=url, status_code=e.response.status_code)
        raise ScrapingError(f"Scraping failed with HTTP status {e.response.status_code} for URL: {url}") from e
    except httpx.RequestError as e:
        logger.error("Scraper network request failed", url=url, error=str(e))
        raise ScrapingError(f"Scraping network request failed for URL {url}: {e}") from e
    except Exception as e:
        logger.error("Unexpected error in scraper", url=url, error=str(e))
        raise ScrapingError(f"Unexpected scraping error for URL {url}: {e}") from e
