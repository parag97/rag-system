"""YAML-backed application configuration."""

import yaml
from pydantic import BaseModel, Field, model_validator


class EmbeddingConfig(BaseModel):
    """Settings for text embedding models.

    Attributes:
        sparse_model_name: Model identifier for sparse embeddings (e.g., BM25).
        vector_model_name: Model identifier for dense embeddings (e.g., sentence-transformers).
    """

    sparse_model_name: str = Field(min_length=1)
    vector_model_name: str = Field(min_length=1)


class QdrantConfig(BaseModel):
    """Connection settings for the Qdrant vector database.

    Attributes:
        host: Qdrant server hostname.
        port: Qdrant HTTP/gRPC port.
        collection_name: Target collection for document vectors.
        top_k: Number of nearest neighbors to return on search.
    """

    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    collection_name: str = Field(min_length=1)
    top_k: int = Field(gt=0)

class ChunkingConfig(BaseModel):
    """Settings for document chunking.

    Attributes:
        chunk_size: Target size in characters for each chunk.
        chunk_overlap: Number of characters to overlap between adjacent chunks.
    """

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        # Ensure chunk_overlap is smaller than chunk_size.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self

class reRankerConfig(BaseModel):
    """Settings for re-ranking search results.

    Attributes:
        cross_encoder_model_name: Model identifier for the cross-encoder used in re-ranking.
        top_k: Number of top search results to return after re-ranking.
    """

    cross_encoder_model_name: str = Field(min_length=1)
    top_k: int = Field(gt=0)

class ContextAssemblerConfig(BaseModel):
    """Settings for assembling retrieved chunks into a context string.

    Attributes:
        max_characters: Maximum total character length for the assembled context.
    """

    max_characters: int = Field(gt=0)

class expanderConfig(BaseModel):
    """Settings for query expansion.

    Attributes:
        expansion_model_name: Model identifier for the query expansion model.
    """

    expansion_window_size: int = Field(gt=0)



class AppConfig(BaseModel):
    """Root configuration loaded from a YAML file.

    Groups all environment-specific settings for embeddings, vector storage,
    and chunking. Parsed once at application startup.

    Attributes:
        embedding: Embedding model settings (dense + sparse).
        qdrant: Vector store connection settings.
        chunking: Document chunking configuration.
        reranker: Re-ranking configuration.
    """

    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    chunking: ChunkingConfig
    reranker: reRankerConfig



def load_config(path: str = "configs/dev.yml") -> AppConfig:
    """Load and validate application config from a YAML file.

    The YAML file should define top-level keys `embedding`, `qdrant`, and
    `chunking` that map to the respective config classes. Pydantic validates
    all required fields and types on construction.

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

if __name__ == "__main__":
    # Example usage: load config and print it.
    config = load_config("configs/dev.yml")