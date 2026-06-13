"""Orchestration for the document ingestion pipeline."""

import logging
import time
from pathlib import Path

from src.chunking.base import Chunker
from src.embeddings.base import DenseEmbeddingService
from src.embeddings.base import SparseEmbeddingService
from src.ingestion.loaders.base import DocumentLoader
from src.ingestion.models import IngestionMetrics, IngestionResult, StageMetrics
from src.vectordb.base import VectorStore
from src.vectordb.models import VectorPoint


logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """Orchestrates loading, chunking, embedding, and vector persistence.

    This class coordinates the ingestion pipeline stages and records
    simple timing metrics for monitoring. It expects pluggable
    implementations for document loading, chunking, embedding, and
    vector storage.
    """

    def __init__(
        self,
        document_loader: DocumentLoader,
        chunker: Chunker,
        vector_embedding_service: DenseEmbeddingService,
        sparse_embedding_service: SparseEmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        # Dependencies wired in from the application bootstrap.
        self._document_loader = document_loader
        self._chunker = chunker
        self._vector_embedding_service = vector_embedding_service
        self._vector_store = vector_store
        self._sparse_embedding_service = sparse_embedding_service

    def ingest(self, source_path: str | Path) -> IngestionResult:
        """Run a single ingestion pipeline for the given source path.

        Steps:
        1. Load the document from disk.
        2. Split into chunks.
        3. Embed chunks (dense + sparse).
        4. Build `VectorPoint`s and upsert to the vector store.

        Returns an `IngestionResult` summarizing counts and timing.
        """
        total_start = time.perf_counter()

        # Load
        stage_start = time.perf_counter()
        document = self._document_loader.load(source_path)
        load_time = time.perf_counter() - stage_start

        # Chunk
        stage_start = time.perf_counter()
        chunks = self._chunker.create_chunks(document)
        chunk_time = time.perf_counter() - stage_start

        # Embed (dense)
        stage_start = time.perf_counter()
        vector_embeddings = self._vector_embedding_service.embed_documents(
            [chunk.text for chunk in chunks]
        )
        embed_time = time.perf_counter() - stage_start

        if len(chunks) != len(vector_embeddings):
            raise ValueError("Number of embeddings does not match number of chunks.")

        # Embed (sparse)
        sparse_embeddings = self._sparse_embedding_service.embed_documents(
            [chunk.text for chunk in chunks]
        )

        # Build VectorPoint objects for upsert; keep ordering consistent.
        points = [
            VectorPoint(
                id=chunk.vector_id,
                vectorEmbedding=vector_embedding,
                SparseEmbedding=sparse_embedding,
                payload=chunk,
            )
            for chunk, vector_embedding, sparse_embedding in zip(
                chunks, vector_embeddings, sparse_embeddings, strict=True
            )
        ]

        # Persist
        stage_start = time.perf_counter()
        self._vector_store.upsert(points)
        upsert_time = time.perf_counter() - stage_start

        # Build result summary
        total_time = time.perf_counter() - total_start
        page_count = len(document.pages)
        chunk_count = len(chunks)
        embedding_count = len(vector_embeddings)
        vector_count = len(points)

        logger.info(
            "Ingested '%s': %d pages, %d chunks, %d vectors in %.3f seconds.",
            document.source_file,
            page_count,
            chunk_count,
            vector_count,
            total_time,
        )

        return IngestionResult(
            document_id=document.document_id,
            source_file=document.source_file,
            page_count=page_count,
            chunk_count=chunk_count,
            embedding_count=embedding_count,
            vector_count=vector_count,
            metrics=IngestionMetrics(
                load=StageMetrics.from_duration(load_time, page_count),
                chunking=StageMetrics.from_duration(chunk_time, chunk_count),
                embedding=StageMetrics.from_duration(embed_time, embedding_count),
                upsert=StageMetrics.from_duration(upsert_time, vector_count),
                total_duration_seconds=total_time,
            ),
        )
