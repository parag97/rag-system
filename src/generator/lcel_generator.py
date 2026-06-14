from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.context.model import Context
from src.generator.base import responseGenerator


class LCELGenerator(responseGenerator):
    def __init__(self, llm):
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

        self.chain = prompt | llm | StrOutputParser()

    def generate_response(self, query: str, context: Context) -> str:
        return self.chain.invoke(
            {
                "query": query,
                "context": context.text,
            }
        )