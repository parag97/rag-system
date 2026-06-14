from abc import ABC, abstractmethod
from src.core.search import SearchResult 
from src.context.model import Context
from src.vectordb.base import VectorStore  


class ContextChunkAssembler(ABC):
    @abstractmethod
    def assembleChunks(self, retrieved_chunks: list[SearchResult]) -> Context:
        """
        Assemble the retrieved search results into a context string that can be used for answering the query.
        :param retrieved_chunks: A list of retrieved text chunks relevant to the query.
        :return: A Context object containing the assembled context and sources.
        """
        pass



class ContextExpander(ABC):

    @abstractmethod
    def expand(
        self,
        chunks: list[SearchResult],
    ) -> list[SearchResult]:
        pass


        
class ContextBuilder(ABC):
    @abstractmethod
    def build(
        self, 
        chunks: list[SearchResult], 
        ) -> Context:
        pass