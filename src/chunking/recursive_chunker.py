from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import Chunker

from src.core.chunk import Chunk

from src.ingestion.document import Document

class RecursiveChunker(Chunker):
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
