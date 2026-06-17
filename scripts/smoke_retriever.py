"""Smoke test for end-to-end retrieval pipeline.

Demonstrates the retrieval stage of RAG:
- Load embeddings and connect to vector store
- Execute semantic search with hybrid (dense + sparse) ranking
- Print top results with scores and metadata

Usage:
    python scripts/smoke_retriever.py
"""

from src.core.config import load_config
from src.embeddings.sentence_transformer_service import (
    SentenceTransformerEmbeddingService,
)
from src.embeddings.bm25_service import BM25EmbeddingService
from src.retrieval.retriever import Retriever
from src.vectordb.qdrant_store import QdrantVectorStore
from src.reranking.CrossEncoderReRanker import CrossEncoderReRanker


def main() -> None:
    """Run retrieval pipeline smoke test."""
    # Load configuration
    config = load_config()

    # Initialize embedding services
    dense_embedder = SentenceTransformerEmbeddingService(
        model_name=config.embedding.vector_model_name,
    )
    sparse_embedder = BM25EmbeddingService(
        model_name=config.embedding.sparse_model_name,
    )

    # Initialize re-ranker
    reranker = CrossEncoderReRanker(
        model_name=config.reranker.cross_encoder_model_name,
        top_k=config.reranker.top_k,
    )

    # Connect to vector store
    vector_store = QdrantVectorStore(
        host=config.qdrant.host,
        port=config.qdrant.port,
        top_k=config.qdrant.top_k,
        collection_name=config.qdrant.collection_name,
    )

    # Wire retriever
    retriever = Retriever(
        dense_embedding_service=dense_embedder,
        sparse_embedding_service=sparse_embedder,
        vector_store=vector_store,
        re_ranker=reranker,
    )

    # Run sample query
    query = "How much money does amazon have?"
    print(f"Query: {query}\n")

    # Retrieve results
    results = retriever.retrieve(query)

    # Display results
    for i, result in enumerate(results, 1):
        print("=" * 77)
        print(f"Result #{i}")
        print(f"Score: {result.score:.4f}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Source: {result.source_file}")
        print()
        print(f"Text: {result.text}")
        print()


if __name__ == "__main__":
    main()

