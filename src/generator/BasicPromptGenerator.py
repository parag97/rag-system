from src.generator.base import promptGenerator
from src.context.model import Context
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

class BasicPromptGenerator(promptGenerator):
    """A basic implementation of the prompt generator that formats the prompt
    by combining the context and the query in a straightforward manner.
    """
    def __init__(self):
        pass

    def generate_prompt(self, context: Context, query: str) -> ChatPromptTemplate:
        """Generate a prompt for the LLM based on the provided context and query.

        Args:
            context (Context): The assembled context relevant to the query.
            query (str): The user's original query.

        Returns:
            ChatPromptTemplate: A chat prompt template representing the prompt to be sent to the LLM.
        """



        RAG_PROMPT = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                            You are a helpful AI assistant.

                            Answer the user's question using ONLY the provided context.

                            If the answer cannot be found in the context, explicitly say:
                            "I could not find the answer in the provided context."

                            Do not make up facts.
                            """.strip(),
                    ),
                    (
                        "human",
                        """
                            Context:
                            {context}

                            Question:
                            {query}
                            """.strip(),
                                    ),
                                ]
                            )
                    


        return RAG_PROMPT