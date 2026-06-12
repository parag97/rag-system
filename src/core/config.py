"""YAML-backed application configuration."""

import yaml
from pydantic import BaseModel, Field, model_validator


class EmbeddingConfig(BaseModel):
    """Settings for the text embedding model.

    Attributes:
        model_name: Hugging Face model ID or local path passed to
            :class:`~src.embeddings.sentence_transformer_service.SentenceTransformerEmbeddingService`.
    """

    sparse_model_name: str = Field(min_length = 1)
    vector_model_name: str = Field(min_length = 1)


class QdrantConfig(BaseModel):
    """Connection settings for the Qdrant vector database.

    Attributes:
        host: Qdrant server hostname.
        port: Qdrant HTTP/gRPC port.
        collection_name: Target collection for document vectors.
    """

    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    collection_name: str = Field(min_length=1)

class ChunkingConfig(BaseModel):
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self

class AppConfig(BaseModel):
    """Root configuration loaded from a YAML file.

    Attributes:
        embedding: Embedding model settings.
        qdrant: Vector store connection settings.
    """

    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    chunking: ChunkingConfig



def load_config(path: str = "configs/dev.yml") -> AppConfig:
    """Load and validate application config from a YAML file.

    Args:
        path: Filesystem path to the YAML config file.

    Returns:
        Parsed :class:`AppConfig` instance.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        pydantic.ValidationError: If required keys are missing or invalid.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return AppConfig(**data)
