"""Google Generative AI LLM backend."""

from langchain_google_genai import ChatGoogleGenerativeAI

from src.llm.base import LLM
from src.core.llm_config import llm_settings


class GoogleStudioGenerator(LLM):
    """Google Generative AI (Gemini) LLM wrapper.

    Provides access to Google's Generative AI models through LangChain,
    configured via environment or pydantic settings.

    Attributes:
        llm: Initialized ChatGoogleGenerativeAI instance.
    """

    def __init__(self) -> None:
        """Initialize the Google Generative AI LLM.

        Uses credentials from llm_settings (typically from environment variables).
        Sets temperature to 0.0 for deterministic, fact-based responses.

        Raises:
            ValueError: If API key is not configured.
        """
        self.llm = ChatGoogleGenerativeAI(
            model=llm_settings.google_model,
            api_key=llm_settings.google_api_key.get_secret_value(),
            temperature=0.0,  # Deterministic responses for RAG
        )

    def get_llm(self):
        """Retrieve the initialized LLM instance.

        Returns:
            ChatGoogleGenerativeAI instance ready for inference.
        """
        return self.llm


