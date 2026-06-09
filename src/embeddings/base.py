from abc import ABC, abstractmethod


class EmbeddingService(ABC):

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        pass