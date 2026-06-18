from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


# ── Ingestion / document management ──────────────────────────────────────────

class IngestRequest(BaseModel):
    """Request body for the manual ingest endpoint."""
    file_path: str


class IngestResponse(BaseModel):
    """Confirmation returned after a successful ingest or re-ingest."""
    document_id: str
    source_file: str
    page_count: int
    chunk_count: int
    vector_count: int


class DocumentEntry(BaseModel):
    """A single document as stored in the registry."""
    source_file: str
    document_id: str
    file_hash: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentEntry]
    total: int


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
