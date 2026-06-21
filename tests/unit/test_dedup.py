import pytest
from unittest.mock import AsyncMock, MagicMock
from core.models import SearchResult
from search.dedup import cosine_similarity, deduplicate_by_url, deduplicate_semantically
from db.embeddings import LocalEmbeddings

def test_cosine_similarity():
    # Test identical vectors
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    assert abs(cosine_similarity([3.0, 4.0], [3.0, 4.0]) - 1.0) < 1e-9
    
    # Test orthogonal vectors
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-9
    
    # Test opposite vectors
    assert abs(cosine_similarity([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-9
    
    # Test zero vectors
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

def test_deduplicate_by_url():
    results = [
        SearchResult(title="Page 1", url="https://example.com/abc", content="content 1"),
        SearchResult(title="Page 2", url="https://example.com/abc/", content="content 2"),
        SearchResult(title="Page 3", url=" HTTPS://example.com/ABC ", content="content 3"),
        SearchResult(title="Page 4", url="https://example.com/def", content="content 4"),
    ]
    
    deduped = deduplicate_by_url(results)
    
    # Should keep Page 1 (first occurrence) and Page 4. Page 2 and Page 3 are normalized duplicates.
    assert len(deduped) == 2
    assert deduped[0].title == "Page 1"
    assert deduped[1].title == "Page 4"

@pytest.mark.asyncio
async def test_deduplicate_semantically():
    # Mock LocalEmbeddings
    mock_embeddings = MagicMock(spec=LocalEmbeddings)
    
    # We will pass 3 documents. 
    # doc 1 and doc 2 will have high similarity (>0.95)
    # doc 3 will have low similarity
    # We set up mock return vectors:
    # vector 1: [1.0, 0.0]
    # vector 2: [0.96, 0.28]  --> cosine sim = 0.96 (exceeds 0.95 threshold)
    # vector 3: [0.0, 1.0]   --> cosine sim = 0.0
    
    vectors = [
        [1.0, 0.0],
        [0.96, 0.28],
        [0.0, 1.0]
    ]
    
    mock_embeddings.aembed_documents = AsyncMock(return_value=vectors)
    
    results = [
        SearchResult(title="Page 1", url="https://url1.com", content="High quality research on A"),
        SearchResult(title="Page 2", url="https://url2.com", content="Extremely similar research on A"),
        SearchResult(title="Page 3", url="https://url3.com", content="Research on a completely different topic B"),
    ]
    
    deduped = await deduplicate_semantically(results, mock_embeddings, threshold=0.95)
    
    # Page 2 should be filtered out because sim is 0.96 > 0.95
    assert len(deduped) == 2
    assert deduped[0].title == "Page 1"
    assert deduped[1].title == "Page 3"
    
    # Check that aembed_documents was called with the correct text lists
    mock_embeddings.aembed_documents.assert_called_once_with([
        "High quality research on A",
        "Extremely similar research on A",
        "Research on a completely different topic B"
    ])

@pytest.mark.asyncio
async def test_deduplicate_semantically_empty_handling():
    mock_embeddings = MagicMock(spec=LocalEmbeddings)
    
    # results with empty content should not be passed to aembed_documents but preserved
    results = [
        SearchResult(title="Page 1", url="https://url1.com", content="some content"),
        SearchResult(title="Page 2", url="https://url2.com", content=""),
        SearchResult(title="Page 3", url="https://url3.com", content="  "),
    ]
    
    mock_embeddings.aembed_documents = AsyncMock(return_value=[[1.0, 0.0]])
    
    deduped = await deduplicate_semantically(results, mock_embeddings)
    
    # Only Page 1 is sent to embed
    mock_embeddings.aembed_documents.assert_called_once_with(["some content"])
    
    # All are preserved
    assert len(deduped) == 3

@pytest.mark.asyncio
async def test_deduplicate_semantically_fallback_on_error():
    mock_embeddings = MagicMock(spec=LocalEmbeddings)
    mock_embeddings.aembed_documents = AsyncMock(side_effect=Exception("Embedding model error"))
    
    results = [
        SearchResult(title="Page 1", url="https://url1.com", content="content 1"),
        SearchResult(title="Page 2", url="https://url2.com", content="content 2"),
    ]
    
    # On error, it should log the error and fall back to the original list
    deduped = await deduplicate_semantically(results, mock_embeddings)
    assert len(deduped) == 2
    assert deduped == results
