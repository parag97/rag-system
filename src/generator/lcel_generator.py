"""LangChain Expression Language-based response generator."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.context.model import Context
from src.generator.base import responseGenerator


class LCELGenerator(responseGenerator):
    """Generate LLM responses using LangChain Expression Language (LCEL).

    This generator combines a chat prompt template, LLM, and output parser
    into a single chain for end-to-end response generation.

    Attributes:
        chain: LCEL chain (prompt | llm | parser) for response generation.
    """

    def __init__(self, llm):
        """Initialize the generator with an LLM backend.

        Creates a reusable chain that pipes together:
        1. Prompt template (formats context + query)
        2. LLM (generates response)
        3. String output parser (extracts text)

        Args:
            llm: Language model instance (e.g., ChatGoogleGenerativeAI).
        """
        # Define the RAG prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a helpful AI assistant.\n"
                        "Answer the user's question using ONLY the provided context.\n"
                        "If the answer cannot be found in the context, say you don't know."
                    ),
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

        # Build the LCEL chain: prompt | llm | parser
        self.chain = prompt | llm | StrOutputParser()

    def generate_response(
        self,
        query: str,
        context: Context,
    ) -> str:
        """Generate an LLM response given context and query.

        Args:
            query: User's natural-language question.
            context: Assembled context with retrieved chunks.

        Returns:
            Generated response text from the LLM.
        """
        response = self.chain.invoke(
            {
                "query": query,
                "context": context.text,
            }
        )
        return response
