from pydantic import BaseModel

from src.core.chunk import Chunk



class VectorPoint(BaseModel):

    id: str

    collection_name: str

    vector: list[float]

    payload: Chunk



class ChunkPayload(BaseModel):
    id: str

    text: str

    source_file: str
    start_page: int | None = None

    end_page: int | None = None

    section_title: str | None = None


class QdrantConfig(BaseModel):
    host: str
    port: int
    collection_name: str








