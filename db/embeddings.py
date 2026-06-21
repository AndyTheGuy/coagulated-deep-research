from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class LocalEmbeddings(Embeddings):
    """Local embeddings service using sentence-transformers all-mpnet-base-v2."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents/texts."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents/texts asynchronously."""
        import asyncio
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Embed a single query string asynchronously."""
        import asyncio
        return await asyncio.to_thread(self.embed_query, text)
