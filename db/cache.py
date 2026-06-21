import sqlite3
import json
import asyncio
import structlog
from typing import Any, Dict, List, Optional
from core.models import SearchResult
from db.embeddings import LocalEmbeddings

logger = structlog.get_logger("deep-research")

def calculate_cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate the cosine similarity between two float vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

class SemanticCache:
    """SQLite-backed local cache layer for search queries and scraped URL content.
    Includes exact match caching and semantic query fallback lookup.
    """
    
    def __init__(
        self, 
        db_path: str = "cache.db", 
        embeddings: Optional[LocalEmbeddings] = None, 
        similarity_threshold: float = 0.90
    ) -> None:
        self.db_path = db_path
        self.embeddings = embeddings or LocalEmbeddings()
        self.similarity_threshold = similarity_threshold
        self._init_db()

    def _init_db(self) -> None:
        """Create standard schema tables if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_text TEXT PRIMARY KEY,
                    embedding TEXT,
                    results_json TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS url_cache (
                    url TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _get_exact_query(self, query: str) -> Optional[List[SearchResult]]:
        """Synchronous exact match lookup in SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT results_json FROM query_cache WHERE query_text = ?", (query,))
            row = cursor.fetchone()
            if row:
                logger.info("Cache hit: exact match", query=query)
                results_data = json.loads(row[0])
                return [SearchResult(**item) for item in results_data]
        return None

    def _get_all_cached_embeddings(self) -> List[Dict[str, Any]]:
        """Retrieve all cached query embeddings for similarity scanning."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT query_text, embedding, results_json FROM query_cache")
            rows = cursor.fetchall()
            cached = []
            for query_text, emb_str, results_json in rows:
                if emb_str:
                    try:
                        emb = json.loads(emb_str)
                        cached.append({
                            "query_text": query_text,
                            "embedding": emb,
                            "results_json": results_json
                        })
                    except Exception as e:
                        logger.warning("Failed to parse cached embedding", query=query_text, error=str(e))
            return cached

    def _set_query_sync(self, query: str, embedding_str: str, results_json: str) -> None:
        """Synchronous save of search query results and embedding."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO query_cache (query_text, embedding, results_json, timestamp)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (query, embedding_str, results_json))
            conn.commit()

    def _get_url_sync(self, url: str) -> Optional[Dict[str, str]]:
        """Synchronous lookup for scraped URL."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title, content FROM url_cache WHERE url = ?", (url,))
            row = cursor.fetchone()
            if row:
                return {"title": row[0], "content": row[1]}
        return None

    def _set_url_sync(self, url: str, title: str, content: str) -> None:
        """Synchronous save of scraped URL content."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO url_cache (url, title, content, timestamp)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (url, title, content))
            conn.commit()

    async def get_query(self, query: str) -> Optional[List[SearchResult]]:
        """Get cached search results for a query (exact or semantic similarity)."""
        # 1. Check exact match
        exact_match = await asyncio.to_thread(self._get_exact_query, query)
        if exact_match is not None:
            return exact_match
            
        # 2. Semantic matching fallback
        logger.info("No exact match; running semantic cache lookup", query=query)
        # Compute query embedding
        query_vector = await self.embeddings.aembed_query(query)
        if not query_vector:
            return None
            
        cached_entries = await asyncio.to_thread(self._get_all_cached_embeddings)
        if not cached_entries:
            return None
            
        best_match = None
        best_score = -1.0
        
        for entry in cached_entries:
            score = calculate_cosine_similarity(query_vector, entry["embedding"])
            if score > best_score:
                best_score = score
                best_match = entry
                
        if best_match and best_score >= self.similarity_threshold:
            logger.info(
                "Cache hit: semantic match", 
                query=query, 
                matched_query=best_match["query_text"], 
                similarity=best_score
            )
            results_data = json.loads(best_match["results_json"])
            return [SearchResult(**item) for item in results_data]
            
        return None

    async def set_query(self, query: str, results: List[SearchResult]) -> None:
        """Cache search results and the query embedding asynchronously."""
        query_vector = await self.embeddings.aembed_query(query)
        embedding_str = json.dumps(query_vector) if query_vector else ""
        results_json = json.dumps([item.model_dump() for item in results])
        
        await asyncio.to_thread(self._set_query_sync, query, embedding_str, results_json)
        logger.info("Successfully cached query results", query=query)

    async def get_url(self, url: str) -> Optional[Dict[str, str]]:
        """Lookup scraped URL content asynchronously."""
        return await asyncio.to_thread(self._get_url_sync, url)

    async def set_url(self, url: str, title: str, content: str) -> None:
        """Cache scraped URL content asynchronously."""
        await asyncio.to_thread(self._set_url_sync, url, title, content)
        logger.info("Successfully cached scraped URL", url=url)
