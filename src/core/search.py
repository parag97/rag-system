from pydantic import BaseModel


class SearchResult(BaseModel):

    chunk_id: str

    score: float

    text: str

    source_file: str

    start_page: int | None = None

    end_page: int | None = None

    section_title: str | None = None