"""Context expansion by adding neighboring chunks from source documents."""

from src.context.base import ContextExpander
from src.core.search import SearchResult


class NeighborContextExpander(ContextExpander):
    """Expands retrieved chunks with adjacent chunks from the same document.

    This context expansion strategy takes retrieved (seed) chunks and includes
    neighboring chunks from the same source document. This provides additional
    context that may be relevant without requiring extra queries.

    For example, with window_size=1, if chunk N is retrieved, this expander
    will also include chunks N-1 and N+1 if they exist.

    Attributes:
        vector_store: Vector store with document chunk metadata.
        window_size: Number of neighbors to include on each side of seed chunk.
    """

    def __init__(self, vector_store, window_size: int = 1):
        """Initialize the context expander.

        Args:
            vector_store: Backend storage with chunk lookup by document and range.
            window_size: Number of neighbor chunks on each side. Defaults to 1.
        """
        self.vector_store = vector_store
        self.window_size = window_size

    def expand(
        self,
        chunks: list[SearchResult],
    ) -> list[SearchResult]:
        """Expand retrieved chunks with adjacent document neighbors.

        For each seed chunk, this method:
        1. Marks it as "retrieved" type.
        2. Queries the vector store for neighboring chunks.
        3. Marks neighbors as "expanded" type.
        4. Deduplicates by chunk_id.

        Args:
            chunks: Retrieved search results to expand.

        Returns:
            Combined list of retrieved and expanded chunks.
        """
        expanded_chunks: list[SearchResult] = []
        seen = set()

        # Process each retrieved chunk and fetch its neighbors
        for chunk in chunks:
            # Mark the seed chunk as retrieved type
            chunk.type = "retrieved"
            chunk.seed_chunk = chunk.chunk_id
            expanded_chunks.append(chunk)
            seen.add(chunk.chunk_id)

            # Skip expansion if chunk lacks positional metadata
            document_id = chunk.source_file
            chunk_index = chunk.chunk_index
            if document_id is None or chunk_index is None:
                continue

            # Calculate neighbor range using window_size
            start_index = max(0, chunk_index - self.window_size)
            end_index = chunk_index + self.window_size + 1

            # Retrieve neighbors from vector store
            neighboring_chunks = self.vector_store.get_chunks_by_range(
                document_id,
                start_index,
                end_index,
            )

            # Add neighbors (deduplicating by chunk_id)
            for neighbor in neighboring_chunks:
                if neighbor.chunk_id in seen:
                    continue
                seen.add(neighbor.chunk_id)

                # Mark as expanded and link to seed
                neighbor.type = "expanded"
                neighbor.seed_chunk = chunk.chunk_id
                expanded_chunks.append(neighbor)

        return expanded_chunks
