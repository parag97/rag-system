"""Smoke test for end-to-end RAG service.

This script demonstrates the complete RAG pipeline:
1. Loading config and initializing all components
2. Setting up retrievers (dense + sparse embeddings)
3. Building context with expansion and assembly
4. Generating responses with an LLM

Usage:
    python scripts/smoke_rag_service.py "Your question here"
    python scripts/smoke_rag_service.py  # Defaults to "What is Python?"
"""

import sys

from src.core.config import load_config
from src.embeddings.sentence_transformer_service import (
    SentenceTransformerEmbeddingService,
)
from src.embeddings.bm25_service import BM25EmbeddingService
from src.retrieval.retriever import Retriever
from src.vectordb.qdrant_store import QdrantVectorStore
from src.reranking.CrossEncoderReRanker import CrossEncoderReRanker
from src.context.expander import NeighborContextExpander
from src.context.ContextAssembler import SimpleContextChunkAssembler
from src.context.BasicContextBuilder import DefaultContextBuilder
from src.llm.google_studio import GoogleStudioGenerator
from src.generator.lcel_generator import LCELGenerator
from src.rag.RAGService import RAGService


def main() -> None:
    """Run the end-to-end RAG pipeline smoke test."""
    # Load configuration from YAML
    config = load_config()

    # Initialize embedding services (dense + sparse)
    dense_embedder = SentenceTransformerEmbeddingService(
        model_name=config.embedding.vector_model_name,
    )
    sparse_embedder = BM25EmbeddingService(
        model_name=config.embedding.sparse_model_name,
    )

    # Initialize re-ranker for result refinement
    reranker = CrossEncoderReRanker(
        model_name=config.reranker.cross_encoder_model_name,
        top_k=config.reranker.top_k,
    )

    # Initialize vector store connection
    vector_store = QdrantVectorStore(
        host=config.qdrant.host,
        port=config.qdrant.port,
        top_k=config.qdrant.top_k,
        collection_name=config.qdrant.collection_name,
    )

    # Wire retriever with embedding and storage services
    retriever = Retriever(
        dense_embedding_service=dense_embedder,
        sparse_embedding_service=sparse_embedder,
        vector_store=vector_store,
        re_ranker=reranker,
    )

    # Build context pipeline: expand neighbors + assemble into context
    context_expander = NeighborContextExpander(
        vector_store,
        window_size=2,
    )
    context_assembler = SimpleContextChunkAssembler(
        max_characters=1500,
    )
    context_builder = DefaultContextBuilder(
        expander=context_expander,
        assembler=context_assembler,
    )

    # Initialize LLM and response generator
    llm = GoogleStudioGenerator().get_llm()
    response_generator = LCELGenerator(llm)

    # Assemble the complete RAG service
    rag_service = RAGService(
        retriever=retriever,
        context_builder=context_builder,
        generator=response_generator,
    )

    # Get query from command-line arguments or use default
    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What is Python?"
    )

    print(f"Query: {query}\n")

    try:
        # Generate answer through the full RAG pipeline
        answer = rag_service.answer(query)
    except Exception as exc:
        print(f"Error during RAG generation: {exc}")
        raise

    # Print the result
    print("Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()

