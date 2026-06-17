from src.context.base import ContextBuilder, ContextExpander, ContextChunkAssembler
from src.context.model import Context
from src.core.search import SearchResult


class DefaultContextBuilder(ContextBuilder):
    """Builds the final context from retrieved search results."""

    def __init__(
        self,
        expander: ContextExpander,
        assembler: ContextChunkAssembler,
    ):
        self.expander = expander
        self.assembler = assembler

    def build(
        self,
        chunks: list[SearchResult],
    ) -> Context:
        """
        Build context in two stages:
        1. Expand retrieved chunks with neighboring chunks.
        2. Assemble the expanded chunks into the final context.
        """
        if self.expander is not None:
            expanded_chunks = self.expander.expand(chunks)
        else:
            expanded_chunks = chunks

        context = self.assembler.assemble_chunks(expanded_chunks)

        return context