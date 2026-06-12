# src/chunking/base.py

from abc import ABC, abstractmethod

from src.core.chunk import Chunk
from src.ingestion.document import Document


class Chunker(ABC):
    @abstractmethod
    def create_chunks(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        pass