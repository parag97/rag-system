from abc import ABC, abstractmethod

from src.core.search import SearchResult
from src.vectordb.models import VectorPoint


class VectorStore(ABC):

    @abstractmethod
    def create_collection(
        self,
        dimension: int
    ) -> None:
        pass

    @abstractmethod
    def upsert(
        self,
        points: list[VectorPoint]
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int
    ) -> list[SearchResult]:
        pass