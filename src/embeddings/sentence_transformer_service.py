from sentence_transformers import SentenceTransformer

from src.embeddings.base import EmbeddingService


class SentenceTransformerEmbeddingService(
    EmbeddingService
):

    def __init__(
        self,
        model_name: str
    ):
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()

    def embed(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        return embeddings.tolist()