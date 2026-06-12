"""Models for vector search results returned to callers."""

from pydantic import BaseModel

from src.core.chunk import Chunk


class SearchResult(BaseModel):
    """A single chunk matched by a similarity search.

    Attributes:
        chunk_id: Identifier in ``{document_id}:{chunk_index}`` form.
        score: Similarity score from the vector store (higher is more similar
            for cosine distance in Qdrant).
        text: Chunk text that was indexed.
        source_file: Path or name of the originating document.
        start_page: First page the chunk spans, if known.
        end_page: Last page the chunk spans, if known.
        section_title: Heading or section label for the chunk, if known.
    """

    chunk_id: str
    score: float
    text: str
    source_file: str
    start_page: int | None = None
    end_page: int | None = None
    section_title: str | None = None

    @classmethod
    def from_chunk_match(cls, chunk: Chunk, score: float) -> "SearchResult":
        """Create a result for a chunk matched by the vector store."""
        return cls(
            chunk_id=chunk.chunk_id,
            score=score,
            text=chunk.text,
            source_file=chunk.source_file,
            start_page=chunk.start_page,
            end_page=chunk.end_page,
            section_title=chunk.section_title,
        )
