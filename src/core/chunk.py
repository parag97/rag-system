"""Text chunk model used during indexing and stored in vector payloads."""

from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    """A slice of a source document prepared for embedding and upsert.

    This model represents the minimal metadata and text required to
    generate embeddings and persist a payload in a vector store.

    Attributes:
        document_id: Parent document identifier.
        vector_id: Deterministic UUID used as the vector point key in the store.
        chunk_index: Zero-based position of this chunk within the document.
        text: Chunk body to embed and retrieve.
        source_file: Path or name of the originating file.
        start_page: First page the chunk spans, if known.
        end_page: Last page the chunk spans, if known.
        section_title: Heading or section label, if known.
    """

    document_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    section_title: str | None = Field(default=None, max_length=255)

    @property
    def chunk_id(self) -> str:
        """Stable string ID combining document and chunk index.

        This human-readable id is used to derive a deterministic UUID
        for vector storage and to make debugging easier.
        """
        return f"{self.document_id}:{self.chunk_index}"

    @property
    def vector_id(self) -> str:
        """Deterministic UUID accepted by Qdrant for this chunk identity.

        The UUID is derived from a namespace UUID and the stable chunk_id
        so repeated runs produce the same vector identifier for a chunk.
        """
        return str(uuid5(NAMESPACE_URL, self.chunk_id))

    @model_validator(mode="after")
    def validate_page_range(self) -> "Chunk":
        # Ensure any provided page range is consistent.
        if (
            self.start_page is not None
            and self.end_page is not None
            and self.start_page > self.end_page
        ):
            raise ValueError("start_page must be less than or equal to end_page.")
        return self
