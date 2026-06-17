from functools import cached_property

from src.core.config import load_config

from src.context.ContextAssembler import (
    SimpleContextChunkAssembler,
)
from src.context.BasicContextBuilder import (
    DefaultContextBuilder,
)
from src.context.expander import (
    NeighborContextExpander,
)

from src.embeddings.sentence_transformer_service import (
    SentenceTransformerEmbeddingService,
)
from src.embeddings.bm25_service import (
    BM25EmbeddingService,
)

from src.generator.lcel_generator import (
    LCELGenerator,
)

from src.llm.google_studio import (
    GoogleStudioGenerator,
)

from src.rag.RAGService import (
    RAGService,
)

from src.reranking.CrossEncoderReRanker import (
    CrossEncoderReRanker,
)

from src.retrieval.retriever import (
    Retriever,
)

from src.vectordb.qdrant_store import (
    QdrantVectorStore,
)

from src.chunking.recursive_chunker import (
    RecursiveChunker,
)

from src.ingestion.loaders.pdf_loader import (
    PDFLoader,
)

from src.ingestion.document_ingestion_service import (
    DocumentIngestionService,
)

from src.ingestion.document_registry import (
    DocumentRegistry,
)

from src.ingestion.document_sync_service import (
    DocumentSyncService,
)

from src.ingestion.watcher_service import (
    WatcherService
)





class ApplicationContainer:
    """
    Responsible for wiring all application dependencies.
    """

    @cached_property
    def config(self):
        return load_config()

    # ============================================================
    # Infrastructure
    # ============================================================

    @cached_property
    def vector_store(self):
        cfg = self.config.qdrant

        return QdrantVectorStore(
            host=cfg.host,
            port=cfg.port,
            top_k=cfg.top_k,
            collection_name=cfg.collection_name,
        )

    # ============================================================
    # Embeddings
    # ============================================================

    @cached_property
    def dense_embedding_service(self):
        return SentenceTransformerEmbeddingService(
            model_name=self.config.embedding.vector_model_name
        )

    @cached_property
    def sparse_embedding_service(self):
        return BM25EmbeddingService(
            model_name=self.config.embedding.sparse_model_name
        )

    # ============================================================
    # Retrieval
    # ============================================================

    @cached_property
    def reranker(self):
        cfg = self.config.reranker

        return CrossEncoderReRanker(
            model_name=cfg.cross_encoder_model_name,
            top_k=cfg.top_k,
        )

    @cached_property
    def retriever(self):
        return Retriever(
            dense_embedding_service=self.dense_embedding_service,
            sparse_embedding_service=self.sparse_embedding_service,
            vector_store=self.vector_store,
            re_ranker=self.reranker,
        )

    # ============================================================
    # Context Building
    # ============================================================






    @cached_property
    def context_builder(self):
        return DefaultContextBuilder(
            expander=NeighborContextExpander(
                vector_store=self.vector_store,
                window_size=self.config.expander.expansion_window_size,
            ),
            assembler=SimpleContextChunkAssembler(
                max_characters=self.config.context_assembler.max_characters,
            ),
        )

    # ============================================================
    # Ingestion
    # ============================================================

    @cached_property
    def document_loader(self):
        return PDFLoader()

    @cached_property
    def chunker(self):
        return RecursiveChunker(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
        )

    @cached_property
    def document_ingestion_service(self):
        return DocumentIngestionService(
            document_loader=self.document_loader,
            chunker=self.chunker,
            vector_embedding_service=self.dense_embedding_service,
            sparse_embedding_service=self.sparse_embedding_service,
            vector_store=self.vector_store,
        )

    @cached_property
    def document_registry(self):
        return DocumentRegistry(
            self.config.ingestion.registry_path
        )

    @cached_property
    def document_sync_service(self):
        return DocumentSyncService(
            ingestion_service=self.document_ingestion_service,
            vector_store=self.vector_store,
            registry=self.document_registry,
            file_ready_max_retries=self.config.ingestion.file_ready_max_retries,
            file_ready_retry_delay_seconds=self.config.ingestion.file_ready_retry_delay_seconds,
        )

    @cached_property
    def watcher_service(self):
        return WatcherService(
            watch_folder=self.config.ingestion.watch_folder,
            sync_service=self.document_sync_service,
        )


    # ============================================================
    # Generation
    # ============================================================

    @cached_property
    def llm(self):
        return GoogleStudioGenerator().get_llm()

    @cached_property
    def generator(self):
        return LCELGenerator(
            llm=self.llm,
        )

    # ============================================================
    # RAG
    # ============================================================

    @cached_property
    def rag_service(self):
        return RAGService(
            retriever=self.retriever,
            context_builder=self.context_builder,
            generator=self.generator,
        )