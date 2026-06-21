import pytest
from core.models import SearchResult
from search.fusion import reciprocal_rank_fusion

def test_reciprocal_rank_fusion_basic():
    # List 1: Doc1 (rank 1), Doc2 (rank 2)
    # List 2: Doc2 (rank 1), Doc3 (rank 2)
    list1 = [
        SearchResult(title="Doc 1", url="https://example.com/doc1", content="content 1"),
        SearchResult(title="Doc 2", url="https://example.com/doc2", content="content 2"),
    ]
    list2 = [
        SearchResult(title="Doc 2", url="https://example.com/doc2", content="content 2"),
        SearchResult(title="Doc 3", url="https://example.com/doc3", content="content 3"),
    ]
    
    # We use a smaller k (e.g. k=2) to make manual calculation verification simple
    # RRF(Doc1) = 1 / (2 + 1) = 1/3 = 0.33333333
    # RRF(Doc2) = 1 / (2 + 2) + 1 / (2 + 1) = 1/4 + 1/3 = 0.25 + 0.33333333 = 0.58333333
    # RRF(Doc3) = 1 / (2 + 2) = 1/4 = 0.25
    
    fused = reciprocal_rank_fusion([list1, list2], k=2)
    
    assert len(fused) == 3
    
    # Sort order should be Doc 2, Doc 1, Doc 3
    assert fused[0].title == "Doc 2"
    assert abs(fused[0].score - 0.58333333) < 1e-6
    
    assert fused[1].title == "Doc 1"
    assert abs(fused[1].score - 0.33333333) < 1e-6
    
    assert fused[2].title == "Doc 3"
    assert abs(fused[2].score - 0.25) < 1e-6

def test_reciprocal_rank_fusion_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []

def test_reciprocal_rank_fusion_single_list():
    list1 = [
        SearchResult(title="Doc A", url="https://example.com/a", content="A"),
        SearchResult(title="Doc B", url="https://example.com/b", content="B"),
    ]
    # RRF(A) = 1 / (10 + 1) = 1/11 ~ 0.090909
    # RRF(B) = 1 / (10 + 2) = 1/12 ~ 0.083333
    fused = reciprocal_rank_fusion([list1], k=10)
    
    assert len(fused) == 2
    assert fused[0].title == "Doc A"
    assert fused[1].title == "Doc B"
    assert abs(fused[0].score - (1.0 / 11.0)) < 1e-6
    assert abs(fused[1].score - (1.0 / 12.0)) < 1e-6

def test_reciprocal_rank_fusion_normalization_and_repeats():
    # Checks that case normalizations and trailing slashes are merged
    list1 = [
        SearchResult(title="Doc 1", url="https://example.com/doc1", content="content 1"),
        SearchResult(title="Doc 2", url="https://example.com/DOC2/", content="content 2"),
    ]
    list2 = [
        SearchResult(title="Doc 2 Alternative", url="https://example.com/doc2", content="alt content"),
        SearchResult(title="Doc 1 Repeat", url="https://example.com/doc1", content="repeat 1"),
    ]
    
    # RRF(doc1) = 1/(10+1) + 1/(10+2) = 1/11 + 1/12 = 0.090909 + 0.083333 = 0.174242
    # RRF(doc2) = 1/(10+2) + 1/(10+1) = 1/12 + 1/11 = 0.174242
    # RRF scores are equal! So tie-breaker kicks in alphabetically by normalized url:
    # "https://example.com/doc1" comes before "https://example.com/doc2"
    
    fused = reciprocal_rank_fusion([list1, list2], k=10)
    
    assert len(fused) == 2
    assert fused[0].url == "https://example.com/doc1"
    # Kept the representative from higher rank (list1 rank 2 vs list2 rank 1 - wait, list2 rank 1 is rank 1, so wait:
    # In list1, DOC2/ has rank 2 (rank_idx = 1).
    # In list2, doc2 has rank 1 (rank_idx = 0).
    # So the best rank for doc2 is rank 1 (from list2).
    # Therefore, the representative SearchResult kept should be from list2, which is "Doc 2 Alternative" with url "https://example.com/doc2"!
    assert fused[1].title == "Doc 2 Alternative"
    assert fused[1].url == "https://example.com/doc2"
