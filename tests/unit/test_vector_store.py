import pytest
from qdrant_client.models import Filter, FieldCondition, MatchValue
from db.vector_store import VectorStore
from db.embeddings import LocalEmbeddings

def test_vector_store_operations():
    """Verify basic storage, semantic search, and metadata filtering on in-memory Qdrant."""
    embeddings = LocalEmbeddings()
    vector_store = VectorStore(embeddings=embeddings)
    
    # Insert test documents
    texts = [
        "SearXNG is a free internet metasearch engine.",
        "Qdrant is a vector database designed for high performance.",
        "Google Gemini is a multimodal AI model."
    ]
    metadatas = [
        {"source": "searxng_doc", "category": "search"},
        {"source": "qdrant_doc", "category": "database"},
        {"source": "gemini_doc", "category": "ai"}
    ]
    vector_store.add_documents(texts, metadatas=metadatas)
    
    # Test semantic query search
    results = vector_store.search("multimodal model by Google", limit=1)
    assert len(results) == 1
    assert "Gemini" in results[0]["content"]
    assert results[0]["metadata"]["source"] == "gemini_doc"
    assert results[0]["score"] > 0.0
    
    # Test filtered search (should restrict matches to 'database' category)
    query_filter = Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="database")
            )
        ]
    )
    filtered_results = vector_store.search(
        "metasearch engine or vector index",
        limit=5,
        query_filter=query_filter
    )
    assert len(filtered_results) == 1
    assert "Qdrant" in filtered_results[0]["content"]
    assert filtered_results[0]["metadata"]["source"] == "qdrant_doc"
