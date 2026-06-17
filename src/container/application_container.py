from functools import cached_property

from src.context.ContextAssembler import SimpleContextChunkAssembler
from src.context.BasicContextBuilder import DefaultContextBuilder
from src.context.expander import NeighborContextExpander

from src.generator.lcel_generator import LCELGenerator

from src.llm.google_studio import GoogleStudioGenerator

from src.rag.RAGService import RAGService

from src.vectordb.qdrant_store import QdrantVectorStore

from src.retrieval.hybrid_retriever import HybridRetriever


class ApplicationContainer:
    """
    Responsible for wiring all application dependencies.
    """

    @cached_property
    def vector_store(self):
        return QdrantVectorStore()

    @cached_property
    def retriever(self):
        return HybridRetriever(
            vector_store=self.vector_store,
        )

    @cached_property
    def context_builder(self):
        return DefaultContextBuilder(
            expander=NeighborContextExpander(
                vector_store=self.vector_store,
                window_size=1,
            ),
            assembler=SimpleContextChunkAssembler(),
        )

    @cached_property
    def llm(self):
        return GoogleStudioGenerator.get_llm()

    @cached_property
    def generator(self):
        return LCELGenerator(
            llm=self.llm,
        )

    @cached_property
    def rag_service(self):
        return RAGService(
            retriever=self.retriever,
            context_builder=self.context_builder,
            generator=self.generator,
        )