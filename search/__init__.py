from search.searxng import search_searxng, SearchError, SearXNGError
from search.ddg import search_ddg, DDGSearchError
from search.scraper import scrape_url, ScrapingError
from search.dedup import deduplicate_by_url, deduplicate_semantically, cosine_similarity

__all__ = [
    "search_searxng", 
    "search_ddg", 
    "scrape_url", 
    "SearchError", 
    "SearXNGError", 
    "DDGSearchError", 
    "ScrapingError",
    "deduplicate_by_url",
    "deduplicate_semantically",
    "cosine_similarity"
]
