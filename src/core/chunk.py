from pydantic import BaseModel, Field
from uuid import uuid4

class Chunk(BaseModel):

    document_id: str
    vector_id: str = str(uuid4())


    chunk_index: int = Field(ge=0)

    text: str

    source_file: str

    start_page: int | None = Field(
        default=None,
        ge=1
    )

    end_page: int | None = Field(
        default=None,
        ge=1
    )

    section_title: str | None = Field(
        default=None,
        max_length=255
    )

    @property
    def chunk_id(self) -> str:
        return f"{self.document_id}:{self.chunk_index}"