"""YAML-backed application configuration."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class EmbeddingConfig(BaseModel):
    sparse_model_name: str = Field(min_length=1)
    vector_model_name: str = Field(min_length=1)


class QdrantConfig(BaseModel):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    collection_name: str = Field(min_length=1)
    top_k: int = Field(gt=0)


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )
        return self


class ReRankerConfig(BaseModel):
    cross_encoder_model_name: str = Field(min_length=1)
    top_k: int = Field(gt=0)


class ContextAssemblerConfig(BaseModel):
    max_characters: int = Field(gt=0)


class ExpanderConfig(BaseModel):
    expansion_window_size: int = Field(gt=0)


class IngestionConfig(BaseModel):
    watch_folder: str = Field(min_length=1)
    registry_path: str = Field(min_length=1)
    file_ready_max_retries: int = Field(gt=0)
    file_ready_retry_delay_seconds: int = Field(gt=0)


class AppConfig(BaseModel):
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    chunking: ChunkingConfig
    reranker: ReRankerConfig
    context_assembler: ContextAssemblerConfig
    expander: ExpanderConfig
    ingestion: IngestionConfig


_config: AppConfig | None = None


def load_config() -> AppConfig:
    global _config

    if _config is not None:
        return _config

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "dev.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    _config = AppConfig.model_validate(data)

    return _config