"""Entry point for the rag-system application."""
from src.chunking.recursive_chunker import RecursiveChunker
from src.core.config import load_config
from src.embeddings.sentence_transformer_service import SentenceTransformerEmbeddingService
from src.ingestion.document_ingestion_service import DocumentIngestionService
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.vectordb.qdrant_store import QdrantVectorStore



def main() -> None:
    """Run the default application entry point.

    This is a placeholder demonstrating the basic imports and structure.
    Extend with actual ingestion or retrieval logic as needed.
    """
    print("Hello from rag-system!")


if __name__ == "__main__":
    main()
