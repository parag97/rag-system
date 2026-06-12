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
    """Orchestrates loading, chunking, embedding, and vector persistence."""

    def __init__(self, document_loader: DocumentLoader, chunker: Chunker, 
        vector_embedding_service: DenseEmbeddingService, sparce_embedding_service: SparseEmbeddingService, 
        vector_store: VectorStore) -> None:
        
        
        self._document_loader = document_loader
        self._chunker = chunker
        self._vector_embedding_service = vector_embedding_service
        self._vector_store = vector_store
        self._sparce_embedding_service = sparce_embedding_service


    def ingest(self, source_path: str | Path) -> IngestionResult:
        total_start = time.perf_counter()

        stage_start = time.perf_counter()
        document = self._document_loader.load(source_path)
        load_time = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        chunks = self._chunker.create_chunks(document)
        chunk_time = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        vector_embeddings = self.__vector_embedding_service.embed_documents(
            [chunk.text for chunk in chunks]
        )
        embed_time = time.perf_counter() - stage_start

        if len(chunks) != len(vector_embeddings):
            raise ValueError("Number of embeddings does not match number of chunks.")
        sparce_embeddings = self._sparce_embedding_service.embed_documents(
            [chunk.text for chunk in chunks]
        )

        points = [
            VectorPoint(id=chunk.vector_id, vectorEmbedding= vector_embeddings, SparseEmbedding = sparce_embeddings, payload=chunk)
            for chunk, vector_embedding, sparce_ in zip(chunks, vector_embeddings, sparce_embeddings, strict=True)
        ]

        stage_start = time.perf_counter()
        self._vector_store.upsert(points)
        upsert_time = time.perf_counter() - stage_start

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
