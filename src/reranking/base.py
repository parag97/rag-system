from abc import ABC, abstractmethod
from src.core.search import SearchResult 
class ReRanker(ABC):
    @abstractmethod
    def re_rank(self, search_results: list[SearchResult], query: str) -> list[SearchResult]:
        """Re-rank search results based on relevance to the query.
        Args:
            search_results: Initial ranked search results to re-rank.
            query: Original natural-language query.
        Returns:
            A new list of SearchResult objects, re-ranked by relevance. 
        """
        pass