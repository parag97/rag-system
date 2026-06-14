from langchain_google_genai import ChatGoogleGenerativeAI
from src.llm.base import LLM
from src.core.llm_config import llm_settings


class GoogleStudioGenerator(LLM):
    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(model=llm_settings.google_model, 
                                          api_key=llm_settings.google_api_key.get_secret_value(), 
                                          temperature=0.0)

    def get_llm(self) -> object:
        return self.llm
    

