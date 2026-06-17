"""Abstract base class for search result re-ranking."""

from abc import ABC, abstractmethod

from src.core.search import SearchResult


class ReRanker(ABC):
    """Abstract base for result re-ranking modules.

    Re-rankers refine initial search results by applying more expensive
    relevance models (e.g., cross-encoders) to improve ranking quality.
    """

    @abstractmethod
    def re_rank(
        self,
        search_results: list[SearchResult],
        query: str,
    ) -> list[SearchResult]:
        """Re-rank search results by relevance to query.

        Args:
            search_results: Initial ranked search results.
            query: Original natural-language query.

        Returns:
            Re-ranked search results, potentially with fewer entries.
        """
        pass
