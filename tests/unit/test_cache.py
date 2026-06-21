import pytest
from unittest.mock import AsyncMock, MagicMock
from core.models import SearchResult
from db.cache import SemanticCache, calculate_cosine_similarity

def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert abs(calculate_cosine_similarity(v1, v2) - 1.0) < 1e-9
    
    v3 = [0.0, 1.0, 0.0]
    assert abs(calculate_cosine_similarity(v1, v3) - 0.0) < 1e-9

    v4 = [1.0, 1.0, 0.0]
    # Cosine similarity between [1,0,0] and [1,1,0] should be 1 / (1 * sqrt(2)) = 0.7071
    assert abs(calculate_cosine_similarity(v1, v4) - 0.70710678) < 1e-4

@pytest.fixture
def mock_embeddings():
    mock_emb = MagicMock()
    mock_emb.aembed_query = AsyncMock()
    return mock_emb

@pytest.mark.asyncio
async def test_semantic_cache_url(tmp_path, mock_embeddings):
    db_file = str(tmp_path / "cache.db")
    cache = SemanticCache(db_path=db_file, embeddings=mock_embeddings)
    
    # Initially should be None
    res = await cache.get_url("https://example.com")
    assert res is None
    
    # Store
    await cache.set_url("https://example.com", "Example Page", "Scraped content text.")
    
    # Retrieve
    res = await cache.get_url("https://example.com")
    assert res is not None
    assert res["title"] == "Example Page"
    assert res["content"] == "Scraped content text."

@pytest.mark.asyncio
async def test_semantic_cache_exact_query(tmp_path, mock_embeddings):
    db_file = str(tmp_path / "cache.db")
    cache = SemanticCache(db_path=db_file, embeddings=mock_embeddings)
    
    results = [
        SearchResult(title="Doc 1", url="https://link1.com", content="Excerpt 1", score=0.9),
        SearchResult(title="Doc 2", url="https://link2.com", content="Excerpt 2", score=0.8)
    ]
    
    mock_embeddings.aembed_query.return_value = [0.1, 0.2, 0.3]
    
    # Store
    await cache.set_query("test query string", results)
    
    # Exact lookup
    cached_res = await cache.get_query("test query string")
    assert cached_res is not None
    assert len(cached_res) == 2
    assert cached_res[0].title == "Doc 1"
    assert cached_res[1].content == "Excerpt 2"
    # Ensure exact lookup didn't need to call embeddings
    assert mock_embeddings.aembed_query.call_count == 1  # Only called during set_query

@pytest.mark.asyncio
async def test_semantic_cache_semantic_query(tmp_path, mock_embeddings):
    db_file = str(tmp_path / "cache.db")
    cache = SemanticCache(db_path=db_file, embeddings=mock_embeddings, similarity_threshold=0.90)
    
    # Set query with a specific embedding
    results = [SearchResult(title="Quantum Computing", url="https://qc.com", content="Qubits rule.")]
    
    # Mock embedding for the stored query: v1 = [1.0, 0.0]
    mock_embeddings.aembed_query.return_value = [1.0, 0.0]
    await cache.set_query("how quantum computer works", results)
    
    # Now check a semantically close query: v2 = [0.95, 0.1]
    # Cosine similarity is 0.95 / (1 * sqrt(0.95^2 + 0.01)) = 0.95 / 0.955 = 0.99
    # This should trigger semantic cache hit since threshold is 0.90
    mock_embeddings.aembed_query.reset_mock()
    mock_embeddings.aembed_query.return_value = [0.95, 0.1]
    
    cached_res = await cache.get_query("explain quantum computers")
    assert cached_res is not None
    assert len(cached_res) == 1
    assert cached_res[0].title == "Quantum Computing"
    
    # Now check a semantically distant query: v3 = [0.0, 1.0] (cosine is 0.0)
    # This should be a cache miss
    mock_embeddings.aembed_query.reset_mock()
    mock_embeddings.aembed_query.return_value = [0.0, 1.0]
    
    cached_miss = await cache.get_query("how to bake sourdough bread")
    assert cached_miss is None
