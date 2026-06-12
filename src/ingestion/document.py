"""Core data model for ingested source documents."""

from pydantic import BaseModel, Field


class Page(BaseModel):
    page_number: int = Field(ge=1)
    text: str


class Document(BaseModel):
    document_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    pages: list[Page]
