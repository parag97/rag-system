from abc import ABC, abstractmethod
from src.context.model import Context


class promptGenerator(ABC):
    @abstractmethod
    def generate_prompt(self, context: Context, query: str):
        """
        Generate a prompt for the LLM based on the provided context and query.
        :param context: The assembled context relevant to the query.
        :param query: The user's original query.
        :return: A string representing the prompt to be sent to the LLM.
        """
        pass
class responseGenerator(ABC):
    @abstractmethod
    def generate_response(self, query: str, context: Context) -> str:
        """
        Generate a response based on the provided context and query.
        :param context: The assembled context relevant to the query.
        :param query: The user's original query.
        :return: A string representing the generated response.
        """
        pass