from qdrant_client import QdrantClient
from qdrant_client.http import models
import os

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "dify_vectors")

client = QdrantClient(url=QDRANT_URL)

def upsert_vectors(vectors: list):
    points = [
        models.PointStruct(
            id=i,
            vector=v["embedding"],
            payload={"text": v["text"]}
        )
        for i, v in enumerate(vectors)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"status": "ok", "count": len(points)}

def search_vectors(query_vector: list, limit: int = 3):
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )
    return [{"text": r.payload["text"], "score": r.score} for r in results]