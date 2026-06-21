import math
from typing import List
import structlog
from core.models import SearchResult
from db.embeddings import LocalEmbeddings

logger = structlog.get_logger("deep-research")

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def deduplicate_by_url(results: List[SearchResult]) -> List[SearchResult]:
    """Perform exact URL-based deduplication, keeping the first occurrence.
    
    Args:
        results: List of SearchResult items.
        
    Returns:
        List of unique SearchResult items.
    """
    seen = set()
    deduped = []
    for r in results:
        normalized = r.url.strip().lower().rstrip("/")
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(r)
    logger.info("URL deduplication complete", original_count=len(results), deduped_count=len(deduped))
    return deduped

async def deduplicate_semantically(
    results: List[SearchResult], 
    embeddings: LocalEmbeddings, 
    threshold: float = 0.95
) -> List[SearchResult]:
    """Perform semantic similarity deduplication on search result contents.
    
    Removes items that have a cosine similarity greater than the specified threshold
    compared to any already-kept item.
    
    Args:
        results: List of SearchResult items.
        embeddings: The LocalEmbeddings service.
        threshold: The similarity threshold (default 0.95).
        
    Returns:
        List of semantically unique SearchResult items.
    """
    if len(results) <= 1:
        return results
        
    # Isolate items with non-empty content for embedding
    valid_items = [r for r in results if r.content and r.content.strip()]
    if not valid_items:
        return results
        
    logger.info("Performing semantic deduplication", count=len(valid_items))
    
    try:
        # Embed all contents in a single batch
        texts = [r.content for r in valid_items]
        vectors = await embeddings.aembed_documents(texts)
        
        kept_results = []
        kept_vectors = []
        
        for item, vector in zip(valid_items, vectors):
            is_duplicate = False
            for kept_vector in kept_vectors:
                sim = cosine_similarity(vector, kept_vector)
                if sim > threshold:
                    logger.info("Found semantically redundant result", title=item.title, url=item.url, similarity=sim)
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_results.append(item)
                kept_vectors.append(vector)
                
        # Add back any items that had empty content to ensure we don't drop them due to lack of text
        empty_content_items = [r for r in results if not r.content or not r.content.strip()]
        
        final_results = kept_results + empty_content_items
        logger.info("Semantic deduplication complete", original_count=len(results), deduped_count=len(final_results))
        return final_results
        
    except Exception as e:
        logger.error("Failed semantic deduplication, falling back to original list", error=str(e))
        return results
