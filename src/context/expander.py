

from src.context.base import ContextExpander
from src.core.search import SearchResult


class NeighborContextExpander(ContextExpander):
    """Context expander that adds neighboring chunks to the retrieved results.

    This implementation takes the initially retrieved chunks and expands the context
    by including adjacent chunks from the same source document. This can help provide
    additional relevant information that may be useful for answering the query.
    """
    def __init__(self, vector_store, window_size: int = 1):
        self.vector_store = vector_store
        self.window_size = window_size

    def expand(
        self,
        chunks: list[SearchResult],
    ) -> list[SearchResult]:
        # based on metadata such as source document and chunk index.
        expanded_chunks: list[SearchResult] = []
        seen = set()
        for chunk in chunks:
            document_id = chunk.source_file
            chunk_index = chunk.chunk_index
            chunk.type = "retrieved"
            chunk.seed_chunk = chunk.chunk_id
            expanded_chunks.append(chunk)
            seen.add(chunk.chunk_id)

            # skip chunks missing positional metadata
            if document_id is None or chunk_index is None:
                continue
            start_index = max(0, chunk_index - self.window_size)
            end_index = chunk_index + self.window_size
            neighboring_chunks = self.vector_store.get_chunks_by_range(document_id, start_index, end_index)

            for nc in neighboring_chunks:
                if nc.chunk_id in seen:
                    continue
                seen.add(nc.chunk_id)
                nc.type = "expanded"
                nc.seed_chunk = chunk.chunk_id
                expanded_chunks.append(nc)
        return expanded_chunks
    