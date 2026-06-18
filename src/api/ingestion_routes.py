"""Document ingestion and management endpoints."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_container
from src.api.schemas import (
    DeleteResponse,
    DocumentEntry,
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
)
from src.container.application_container import ApplicationContainer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest or re-ingest a document by file path",
)
async def ingest_document(
    request: IngestRequest,
    container: ApplicationContainer = Depends(get_container),
):
    """Trigger ingestion for a single file.

    - If the file is new, it is loaded, chunked, embedded, and indexed.
    - If the file already exists and is unchanged, the call is a no-op and
      returns the existing registry entry.
    - If the file has changed, it is re-indexed.
    """
    file_path = Path(request.file_path)

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {request.file_path}",
        )

    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a file: {request.file_path}",
        )

    try:
        result = container.document_ingestion_service.ingest(file_path)
    except Exception as exc:
        logger.exception("Ingestion failed for %s", request.file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return IngestResponse(
        document_id=result.document_id,
        source_file=result.source_file,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        vector_count=result.vector_count,
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
async def list_documents(
    container: ApplicationContainer = Depends(get_container),
):
    """Return all documents currently tracked in the registry."""
    entries = container.document_registry.list_entries()

    documents = [
        DocumentEntry(
            source_file=source_path,
            document_id=entry.document_id,
            file_hash=entry.file_hash,
        )
        for source_path, entry in entries.items()
    ]

    return DocumentListResponse(
        documents=documents,
        total=len(documents),
    )


@router.get(
    "/{document_id}",
    response_model=DocumentEntry,
    summary="Get a single document by document_id",
)
async def get_document(
    document_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Look up a document's registry entry by its document_id."""
    entries = container.document_registry.list_entries()

    match = next(
        (
            DocumentEntry(
                source_file=source_path,
                document_id=entry.document_id,
                file_hash=entry.file_hash,
            )
            for source_path, entry in entries.items()
            if entry.document_id == document_id
        ),
        None,
    )

    if match is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    return match


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    summary="Delete a document from the index and registry",
)
async def delete_document(
    document_id: str,
    container: ApplicationContainer = Depends(get_container),
):
    """Remove a document's vectors from Qdrant and its entry from the registry.

    Returns `deleted: false` (HTTP 404) if the document_id is not found.
    """
    deleted = container.document_sync_service.delete_by_document_id(document_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    return DeleteResponse(document_id=document_id, deleted=True)
