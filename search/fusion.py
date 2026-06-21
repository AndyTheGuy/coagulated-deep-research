from typing import List, Dict
import structlog
from core.models import SearchResult

logger = structlog.get_logger("deep-research")

def reciprocal_rank_fusion(
    results_lists: List[List[SearchResult]], 
    k: int = 60
) -> List[SearchResult]:
    """Perform Reciprocal Rank Fusion (RRF) to merge and rank search results.
    
    Formula: RRF(d) = sum_{m in M} (1 / (k + r_m(d)))
    where r_m(d) is the rank (1-indexed) of document d in list m.
    
    Args:
        results_lists: A list of lists of SearchResult items to merge.
        k: A constant parameter for RRF (default 60).
        
    Returns:
        List of merged and ranked SearchResult items.
    """
    if not results_lists:
        return []
        
    logger.info("Executing Reciprocal Rank Fusion", num_lists=len(results_lists))
    
    rrf_scores: Dict[str, float] = {}
    best_results: Dict[str, SearchResult] = {}
    best_ranks: Dict[str, int] = {} # Keep track of the lowest rank index seen for each URL
    
    for list_idx, results in enumerate(results_lists):
        for rank_idx, item in enumerate(results):
            rank = rank_idx + 1 # 1-indexed
            normalized_url = item.url.strip().lower().rstrip("/")
            
            # Update RRF score
            current_score = rrf_scores.get(normalized_url, 0.0)
            rrf_scores[normalized_url] = current_score + (1.0 / (k + rank))
            
            # Track the best representative SearchResult (highest rank / lowest rank_idx)
            if normalized_url not in best_ranks or rank < best_ranks[normalized_url]:
                best_ranks[normalized_url] = rank
                best_results[normalized_url] = item
                
    # Assign the calculated RRF score to the score field of each kept SearchResult
    fused_results = []
    for normalized_url, score in rrf_scores.items():
        best_item = best_results[normalized_url]
        # Create a new SearchResult with updated score to avoid side-effects on original objects
        fused_item = SearchResult(
            title=best_item.title,
            url=best_item.url,
            content=best_item.content,
            score=score,
            engine=best_item.engine
        )
        fused_results.append(fused_item)
        
    # Sort descending by RRF score, with deterministic tie-breaking on normalized URL
    fused_results.sort(key=lambda r: (-r.score, r.url.strip().lower().rstrip("/")))
    
    logger.info("Reciprocal Rank Fusion complete", original_count=sum(len(l) for l in results_lists), fused_count=len(fused_results))
    return fused_results
