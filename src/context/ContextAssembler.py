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

    def _assemble_chunk_with_metadata(self, chunks: list[SearchResult], remaining_length: int) -> tuple[str, int, list[SearchResult]]:
        
        
        """Helper function to assemble a chunk's text with its metadata."""
    

        if not chunks:
            return "", 0, []
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.chunk_index if c.chunk_index is not None else -1,
        )
        # Use the first chunk's metadata as representative for the group
        context_str = f'''
    #####################################################################
    # Next chunk with metadata
    #####################################################################
    '''
        current_document_id = None
        current_page = None
        current_section_title = None
        seen = set()
        context_chunks = []
        for each_chunk in sorted_chunks:
            temp_context = str()
            if each_chunk.chunk_id in seen:
                continue
            seen.add(each_chunk.chunk_id)
            
            if current_document_id is None or current_document_id != each_chunk.source_file:
                current_document_id = each_chunk.source_file
                if(current_document_id is not None):
                    temp_context += f"Document: {current_document_id}\n"

            if current_page is None or current_page != each_chunk.start_page:
                current_page = each_chunk.start_page
                if(current_page is not None):
                    temp_context += f"Page: {current_page}\n"

            if current_section_title is None or current_section_title != each_chunk.section_title:
                current_section_title = each_chunk.section_title
                if(current_section_title is not None):
                    temp_context += f"Section: {current_section_title}\n"

            temp_context += f"Text: {each_chunk.text}\n"
            if(temp_context+context_str).__len__() > remaining_length:
                break
            context_str += temp_context
            context_chunks.append(each_chunk)

        return context_str, len(context_str), context_chunks


    def assembleChunks(self, retrieved_chunks: list[SearchResult]) -> Context:
        """Combine retrieved chunks into a single context string."""

        
        # Order chunks by their original retrieval score (highest first)
        seed_chunks = [chunk for chunk in retrieved_chunks if chunk.type == "retrieved"]
        expanded_chunks = [chunk for chunk in retrieved_chunks if chunk.type == "expanded"]
        ordered_seed_chunks = sorted(seed_chunks, key=lambda c: c.score, reverse=True)
        remaining_length = self.max_characters

        # Concatenate chunk texts with newlines, ensuring we don't exceed the max character limit
        context_str = ""
        length = 0
        seen = set()
        context_chunks: list[SearchResult] = []
        for chunk in ordered_seed_chunks:
            # Skip duplicate chunks based on chunk_id
            if chunk.chunk_id in seen:
                continue    
            seen.add(chunk.chunk_id)
            expanded_neighbors = [c for c in expanded_chunks if c.seed_chunk == chunk.chunk_id]
            all_chunks = expanded_neighbors + [chunk]
            one_chunk_context, one_chunk_length, one_context_chunks = self._assemble_chunk_with_metadata(all_chunks, remaining_length)
            length += one_chunk_length
            if length > remaining_length:
                continue  # Skip this chunk if it exceeds the remaining length
            context_str += one_chunk_context        
            context_chunks.extend(one_context_chunks)
            remaining_length -= one_chunk_length
    
        result = Context(text=context_str.strip(), sources=context_chunks)

        return result
    