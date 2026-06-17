
"""Orchestration layer combining retrieval, context building, and generation."""

from src.context.base import ContextBuilder
from src.generator.base import responseGenerator
from src.retrieval.retriever import Retriever


class RAGService:
    """End-to-end RAG orchestration combining retrieval, context, and generation.

    This service coordinates the three core stages of a retrieval-augmented
    generation pipeline:
    1. **Retrieval**: Embed and find relevant chunks for the query.
    2. **Context Building**: Expand and assemble chunks into unified context.
    3. **Generation**: Use the context and query to generate an LLM response.

    Attributes:
        retriever: Retrieval service for semantic search over indexed documents.
        context_builder: Service to expand and assemble retrieved chunks.
        generator: LLM-based response generator.
    """

    def __init__(
        self,
        retriever: "Retriever",
        context_builder: "ContextBuilder",
        generator: "responseGenerator",
    ) -> None:
        """Initialize RAG service with its three core components.

        Args:
            retriever: Handles semantic search and chunk retrieval.
            context_builder: Expands and assembles chunks into context.
            generator: Generates LLM responses from query + context.
        """
        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator

    def answer(self, query: str) -> str:
        """Generate an LLM response to a user query using retrieval and context.

        This method executes the full RAG pipeline:
        - Retrieve relevant chunks matching the query.
        - Expand chunks with neighbors and assemble into context.
        - Generate response using the context and original query.

        Args:
            query: User's natural-language question or prompt.

        Returns:
            LLM-generated response grounded in retrieved context.

        Raises:
            ValueError: If query is empty or any stage fails.
        """
        # Step 1: Retrieve documents relevant to the query
        retrieved_chunks = self.retriever.retrieve(query)

        # Step 2: Expand and assemble chunks into a cohesive context
        context = self.context_builder.build(retrieved_chunks)

        # Step 3: Generate response using context and query
        response = self.generator.generate_response(query=query, context=context)

        return response
