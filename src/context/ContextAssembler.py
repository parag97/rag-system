from src.context.base import ContextChunkAssembler
from src.core.search import SearchResult
from src.context.model import Context


class SimpleContextChunkAssembler(ContextChunkAssembler):
    """A simple context assembler that concatenates retrieved chunks.

    This implementation takes a list of retrieved text chunks and combines
    them into a single string, separated by newlines. It does not perform
    any additional processing or formatting.
    """
    def __init__(self, max_characters: int = 12000):
        """Initialize the assembler with a maximum character limit for the context."""

        self.max_characters = max_characters



    def assembleChunks(self, retrieved_chunks: list[SearchResult]) -> Context:
        """Combine retrieved chunks into a single context string."""

        
        # Order chunks by their original retrieval score (highest first)
        ordered_chunks = sorted(retrieved_chunks, key=lambda c: c.score, reverse=True)



        # Concatenate chunk texts with newlines, ensuring we don't exceed the max character limit
        length = 0
        context_str = ""
        current_document_id = None
        current_page = None
        current_section_title = None
        seen = set()
        context_chunks = []
        for chunk in ordered_chunks:
            # Skip duplicate chunks based on chunk_id
            if chunk.chunk_id in seen:
                continue    
            seen.add(chunk.chunk_id)
            chunk_length = len(chunk.text) + 1  # +1 for the newline character

            temp_context = ""
            if current_document_id is None or current_document_id != chunk.source_file:
                current_document_id = chunk.source_file
                if(current_document_id is not None):
                    temp_context += f"Document: {current_document_id}\n"
                    chunk_length += len(f"Document: {current_document_id}\n")

            if current_page is None or current_page != chunk.start_page:
                current_page = chunk.start_page
                if(current_page is not None):
                    temp_context += f"Page: {current_page}\n"
                    chunk_length += len(f"Page: {current_page}\n")

            if current_section_title is None or current_section_title != chunk.section_title:
                current_section_title = chunk.section_title
                if(current_section_title is not None):
                    temp_context += f"Section: {current_section_title}\n"
                    chunk_length += len(f"Section: {current_section_title}\n")

            if length + chunk_length > self.max_characters:
                break

            context_chunks.append(chunk)
            context_str += temp_context + chunk.text + "\n"
            
            length += chunk_length

        result = Context(text=context_str.strip(), sources=context_chunks)

        return result
    