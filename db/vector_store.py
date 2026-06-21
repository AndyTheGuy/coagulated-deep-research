import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
from db.embeddings import LocalEmbeddings

class VectorStore:
    """Wrapper class for managing an in-memory Qdrant database."""
    
    def __init__(self, embeddings: Optional[LocalEmbeddings] = None) -> None:
        self.client = QdrantClient(location=":memory:")
        self.embeddings = embeddings or LocalEmbeddings()
        self.collection_name = "research_documents"
        self._setup_collection()

    def _setup_collection(self) -> None:
        """Create the collection with 768-d cosine distance config."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Embed list of texts and upsert points to Qdrant collection."""
        if not texts:
            return
            
        embeddings = self.embeddings.embed_documents(texts)
        points = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            metadata["content"] = text
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=metadata
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        query_filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search query on collection."""
        query_vector = self.embeddings.embed_query(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit
        )
        return [
            {
                "content": hit.payload.get("content", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "content"},
                "score": hit.score
            }
            for hit in results.points
        ]

    async def aadd_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Embed list of texts and upsert points to Qdrant collection asynchronously."""
        import asyncio
        await asyncio.to_thread(self.add_documents, texts, metadatas)

    async def asearch(
        self,
        query: str,
        limit: int = 5,
        query_filter: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search query on collection asynchronously."""
        import asyncio
        return await asyncio.to_thread(self.search, query, limit, query_filter)
