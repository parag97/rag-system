"""Basic prompt generator for RAG queries."""

from src.generator.base import promptGenerator
from src.context.model import Context
from langchain_core.prompts import ChatPromptTemplate


class BasicPromptGenerator(promptGenerator):
    """Simple prompt generator combining context and query into a chat template.

    This generator creates a basic RAG prompt with a system message instructing
    the LLM to answer using only provided context, and a user message containing
    the context and question.
    """

    def __init__(self) -> None:
        """Initialize the basic prompt generator."""
        pass

    def generate_prompt(
        self,
        context: Context,
        query: str,
    ) -> ChatPromptTemplate:
        """Generate a chat prompt template for the LLM.

        The prompt instructs the LLM to:
        - Answer using ONLY the provided context
        - Explicitly state when information is not found
        - Not fabricate facts

        Args:
            context: Assembled context with retrieved chunks.
            query: User's original natural-language question.

        Returns:
            ChatPromptTemplate ready for LLM invocation.
        """
        # Define the system instructions for the LLM
        system_instruction = (
            "You are a helpful AI assistant.\n\n"
            "Answer the user's question using ONLY the provided context.\n\n"
            "If the answer cannot be found in the context, explicitly say:\n"
            '"I could not find the answer in the provided context."\n\n'
            "Do not make up facts."
        )

        # Create the chat prompt template with system and user messages
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_instruction,
                ),
                (
                    "human",
                    (
                        "Context:\n{context}\n\n"
                        "Question:\n{query}"
                    ),
                ),
            ]
        )

        return rag_prompt
