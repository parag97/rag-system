from pydantic import BaseModel
import yaml


class EmbeddingConfig(BaseModel):
    model_name: str


class QdrantConfig(BaseModel):
    host: str
    port: int
    collection_name: str


class AppConfig(BaseModel):
    embedding: EmbeddingConfig
    qdrant: QdrantConfig


def load_config(
    path: str = "configs/dev.yml"
) -> AppConfig:

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return AppConfig(**data)