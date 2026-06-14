from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    google_api_key: SecretStr
    google_model: str = "gemini-2.5-flash"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_llm_settings() -> LLMSettings:
    return LLMSettings() # type: ignore


llm_settings = get_llm_settings()