from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def get_llm(self) -> object:
        """
        Return the underlying language model.
        :return: The language model instance.
        """
        pass