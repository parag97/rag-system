from src.core.search import SearchResult
from src.embeddings.base import EmbeddingService
from src.vectordb.base import VectorStore


class Retriever:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:

        query_vector = self.embedding_service.embed(
            [query]
        )[0]

        return self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
        )