


class RAGService:
    def __init__(self, retriever, context_builder, generator,):

        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator

    def answer(self, query: str) -> str:

        retrieved_chunks = self.retriever.retrieve(query)

        context = self.context_builder.build(retrieved_chunks)

        return self.generator.generate_response(query=query, context=context)
    

    