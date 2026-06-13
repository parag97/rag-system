"""Core data model for ingested source documents.

These lightweight Pydantic models capture the minimal representation of
an input document and its pages as produced by loaders (PDF, etc.).
"""

from pydantic import BaseModel, Field


class Page(BaseModel):
    """A single page extracted from a source document.

    Attributes:
        page_number: 1-based page index within the source file.
        text: Full text extracted from the page.
    """

    page_number: int = Field(ge=1)
    text: str


class Document(BaseModel):
    """Top-level document container returned by loaders.

    Attributes:
        document_id: Deterministic identifier for the document.
        source_file: Path or name of the original file on disk.
        pages: List of `Page` objects in reading order.
    """

    document_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    pages: list[Page]
