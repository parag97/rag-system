from sentence_transformers import CrossEncoder

from src.reranking.base import ReRanker

from src.core.search import SearchResult

class CrossEncoderReRanker(ReRanker):
    """Re-ranker using a cross-encoder model.

    This implementation uses a cross-encoder to score the relevance of
    retrieved chunks against the query. It returns the top-k most relevant
    chunks based on these scores.
    """

    def __init__(self, model_name: str, top_k: int):
        self.model_name = model_name
        self.top_k = top_k

        # Initialize the cross-encoder model 
        self.model = CrossEncoder(model_name)

    def re_rank(self, search_results: list[SearchResult], query: str) -> list[SearchResult]:
        """Re-rank search results based on relevance to the query.

        Args:
            search_results: Initial ranked search results to re-rank.
            query: Original natural-language query.
        Returns:
            A list of the top-k most relevant search results after re-ranking.
        """
        # Compute relevance scores for each chunk using the cross-encoder
        scores = self.model.predict([(query, search_result.text) for search_result in search_results])



        # Pair each search result with its score and sort by score in descending order
        scored_results = sorted(zip(search_results, scores), key=lambda x: x[1], reverse=True)
        for each_result, each_score in scored_results:
            each_result.score = each_score  # Update the score in the SearchResult object
            each_result.text = each_result.text + f" (Re-ranked score: {each_score:.4f})"  # Append the score to the text for demonstration
        scored_results = [result for result, score in scored_results[:self.top_k]]
        return scored_results