"""Recursive chunker implementation using a text splitter.

This module provides `RecursiveChunker`, an implementation of the
`Chunker` protocol that uses `RecursiveCharacterTextSplitter` to break
page text into overlapping chunks suitable for embedding and indexing.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import Chunker

from src.core.chunk import Chunk

from src.ingestion.document import Document


class RecursiveChunker(Chunker):
    """Chunker using recursive character splitting.

    Uses `RecursiveCharacterTextSplitter` to produce overlapping text
    chunks while preserving natural breakpoints where possible.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size."
            )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


    def create_chunks(self, document: Document) -> list[Chunk]:
        """Split document pages into chunks.

        Iterates pages in order and uses the configured splitter to create
        chunk texts. Each produced `Chunk` includes page metadata and a
        sequential `chunk_index` across the whole document.
        """

        chunks: list[Chunk] = []
        chunk_index = 0

        for page in document.pages:
            split_texts = self._splitter.split_text(page.text)

            for chunk_text in split_texts:
                chunks.append(
                    Chunk(
                        document_id=document.document_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        source_file=document.source_file,
                        start_page=page.page_number,
                        end_page=page.page_number,
                    )
                )
                chunk_index += 1

        return chunks
