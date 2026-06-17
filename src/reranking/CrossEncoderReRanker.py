"""Cross-encoder based re-ranking for search results."""

from sentence_transformers import CrossEncoder

from src.reranking.base import ReRanker
from src.core.search import SearchResult


class CrossEncoderReRanker(ReRanker):
    """Re-rank search results using a cross-encoder model.

    Cross-encoders directly score the relevance of query-chunk pairs,
    often outperforming separate dual-encoder dense and sparse methods.
    This re-ranker takes initial retrieval results and re-scores them
    for more accurate top-k selection.

    Attributes:
        model_name: HuggingFace model identifier for the cross-encoder.
        top_k: Number of top results to retain after re-ranking.
        model: Loaded CrossEncoder instance.
    """

    def __init__(self, model_name: str, top_k: int):
        """Initialize the cross-encoder re-ranker.

        Args:
            model_name: HuggingFace model ID (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2').
            top_k: Number of top-ranked results to keep.
        """
        self.model_name = model_name
        self.top_k = top_k

        # Load the cross-encoder model (downloaded from HuggingFace)
        self.model = CrossEncoder(model_name)

    def re_rank(
        self,
        search_results: list[SearchResult],
        query: str,
    ) -> list[SearchResult]:
        """Re-rank search results by query-chunk relevance.

        Uses the cross-encoder to score each query-chunk pair, updates
        relevance scores, and returns the top-k most relevant results.

        Args:
            search_results: Initial ranked search results.
            query: Original natural-language query.

        Returns:
            Top-k search results after re-ranking by cross-encoder score.
        """
        # Prepare query-chunk pairs for cross-encoder scoring
        query_chunk_pairs = [
            (query, result.text)
            for result in search_results
        ]

        # Compute relevance scores for each pair
        relevance_scores = self.model.predict(query_chunk_pairs)

        # Pair results with their scores and sort by score (descending)
        scored_results = sorted(
            zip(search_results, relevance_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Update SearchResult objects with new scores and keep top-k
        re_ranked_results = []
        for result, score in scored_results[: self.top_k]:
            result.score = score
            re_ranked_results.append(result)

        return re_ranked_results
