from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams
)
from src.vectordb.base import VectorStore
from src.core.search import SearchResult
from src.vectordb.models import VectorPoint
from qdrant_client.models import PointStruct


class QdrantVectorStore(VectorStore):

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
    ):
        self.collection_name = collection_name

        self.client = QdrantClient(
            host=host,
            port=port,
        )




    def create_collection(
        self,
        dimension: int
    ) -> None:

        self.client.create_collection(

            collection_name = self.collection_name,
            vectors_config = VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
    


    def upsert(
        self,
        points: list[VectorPoint]
        ) -> None:

        qdrant_points = [
            
            PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload.model_dump()
            )

            for point in points
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=qdrant_points
        )



    def search(
        self,
        query_vector: list[float],
        top_k: int
    ) -> list[SearchResult]:

        resp = self.client.query_points(
            collection_name = self.collection_name,
            query = query_vector,
            limit = top_k
            )
        result = []

        for each_point in resp.points:
            
            payload = each_point.payload

            one_point_data = SearchResult(
                chunk_id = f"{payload['document_id']}:{payload['chunk_index']}",
                score = each_point.score,
                text = payload["text"],
                source_file = payload["source_file"],
                start_page = payload.get("start_page"),
                end_page = payload.get("end_page"),
                section_title = payload.get("section_title"),
            )
            result.append(one_point_data)

        return result    
        