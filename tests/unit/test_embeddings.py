import pytest
from db.embeddings import LocalEmbeddings

def test_embeddings_generation():
    """Verify that LocalEmbeddings produces correct shape vector representations."""
    # Initialize LocalEmbeddings (this will download model on first execution, but it should be fast or cached)
    embeddings = LocalEmbeddings()
    
    # Test single query embedding
    query_text = "What is post-quantum cryptography?"
    query_vector = embeddings.embed_query(query_text)
    
    assert isinstance(query_vector, list)
    assert len(query_vector) == 768
    assert all(isinstance(val, float) for val in query_vector)
    
    # Test multiple documents embedding
    doc_texts = [
        "Quantum computing uses qubits.",
        "Symmetric encryption keys are highly resistant to quantum threats."
    ]
    doc_vectors = embeddings.embed_documents(doc_texts)
    
    assert isinstance(doc_vectors, list)
    assert len(doc_vectors) == 2
    assert len(doc_vectors[0]) == 768
    assert len(doc_vectors[1]) == 768
    assert all(isinstance(val, float) for val in doc_vectors[0])
