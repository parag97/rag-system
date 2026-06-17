"""Context assembly for combining retrieved chunks into a unified prompt."""

from src.context.base import ContextChunkAssembler
from src.core.search import SearchResult
from src.context.model import Context

# Metadata separator line for visual clarity in assembled context
_CHUNK_SEPARATOR = (
    "\n" + "=" * 77 + "\n"
)


class SimpleContextChunkAssembler(ContextChunkAssembler):
    """Concatenates and formats retrieved chunks into a unified context string.

    This assembler takes retrieved search results and their expanded neighbors,
    formats them with metadata (document, page, section), and combines them
    into a single prompt-friendly context string with a character limit.

    Attributes:
        max_characters: Maximum total characters allowed in assembled context.
    """

    def __init__(self, max_characters: int = 12000) -> None:
        """Initialize the assembler with a maximum character limit.

        Args:
            max_characters: Character budget for the final context. Defaults to 12000.
        """
        self.max_characters = max_characters

    def _assemble_chunk_with_metadata(
        self,
        chunks: list[SearchResult],
        remaining_length: int,
    ) -> tuple[str, int, list[SearchResult]]:
        """Format a chunk and its neighbors with metadata headers.

        Sorts chunks by index, adds metadata headers (document, page, section),
        and concatenates until the remaining character budget is exhausted.

        Args:
            chunks: SearchResult chunks to format (primary + neighbors).
            remaining_length: Character budget still available for assembly.

        Returns:
            Tuple of (formatted_text, length_used, chunks_included).
        """
        # Early exit if no chunks to process
        if not chunks:
            return "", 0, []

        # Sort neighbors by chunk index for logical presentation
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.chunk_index if c.chunk_index is not None else -1,
        )

        # Initialize context with separator header
        context_str = _CHUNK_SEPARATOR

        # Track metadata state to emit headers only when values change
        current_document_id = None
        current_page = None
        current_section_title = None

        # Track which chunks we've already added (deduplication)
        seen = set()
        context_chunks = []

        # Process each chunk in sorted order
        for chunk in sorted_chunks:
            # Skip duplicates by chunk_id
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)

            temp_context = ""

            # Emit document header if source changed
            if (
                current_document_id is None
                or current_document_id != chunk.source_file
            ):
                current_document_id = chunk.source_file
                if current_document_id is not None:
                    temp_context += f"Document: {current_document_id}\n"

            # Emit page header if page changed
            if current_page is None or current_page != chunk.start_page:
                current_page = chunk.start_page
                if current_page is not None:
                    temp_context += f"Page: {current_page}\n"

            # Emit section header if section changed
            if (
                current_section_title is None
                or current_section_title != chunk.section_title
            ):
                current_section_title = chunk.section_title
                if current_section_title is not None:
                    temp_context += f"Section: {current_section_title}\n"

            # Add chunk text
            temp_context += f"Text: {chunk.text}\n"

            # Check if adding this chunk would exceed budget
            total_length = len(context_str) + len(temp_context)
            if total_length > remaining_length:
                break

            # Commit this chunk to the context
            context_str += temp_context
            context_chunks.append(chunk)

        return context_str, len(context_str), context_chunks

    def assemble_chunks(self, retrieved_chunks: list[SearchResult]) -> Context:
        """Combine retrieved and expanded chunks into a single context string.

        This method prioritizes seed (retrieved) chunks by score, includes
        their expanded neighbors, and formats everything with metadata
        until the character budget is exhausted.

        Args:
            retrieved_chunks: SearchResults marked as "retrieved" or "expanded".

        Returns:
            Context object containing formatted text and source chunk references.
        """
        # Separate retrieved chunks (primary results) from expanded (neighbors)
        seed_chunks = [
            chunk
            for chunk in retrieved_chunks
            if chunk.type == "retrieved"
        ]
        expanded_chunks = [
            chunk
            for chunk in retrieved_chunks
            if chunk.type == "expanded"
        ]

        # Sort seed chunks by relevance (highest score first)
        ordered_seed_chunks = sorted(
            seed_chunks,
            key=lambda c: c.score,
            reverse=True,
        )

        # Initialize accumulation variables
        context_str = ""
        total_length = 0
        seen = set()
        context_chunks: list[SearchResult] = []

        # Remaining character budget for assembly
        remaining_length = self.max_characters

        # Process each seed chunk with its neighbors
        for chunk in ordered_seed_chunks:
            # Skip duplicates
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)

            # Gather neighbors for this seed chunk
            neighbors = [
                c for c in expanded_chunks if c.seed_chunk == chunk.chunk_id
            ]
            all_related_chunks = neighbors + [chunk]

            # Format this chunk group with metadata
            chunk_context, chunk_length, chunks_included = (
                self._assemble_chunk_with_metadata(
                    all_related_chunks,
                    remaining_length,
                )
            )

            # Update cumulative length
            total_length += chunk_length

            # Skip this chunk if it would exceed budget
            if total_length > self.max_characters:
                continue

            # Add formatted chunk group to final context
            context_str += chunk_context
            context_chunks.extend(chunks_included)

            # Update remaining budget
            remaining_length -= chunk_length

        # Return assembled context with source references
        return Context(
            text=context_str.strip(),
            sources=context_chunks,
        )
    