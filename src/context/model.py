from pydantic import BaseModel

from src.core.search import SearchResult


class Context(BaseModel):
    """Assembled context string for a query, created by `ContextAssembler`."""

    text: str
    sources: list[SearchResult]