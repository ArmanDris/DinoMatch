import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import json

def upload_to_qdrant():
    client = QdrantClient("http://localhost:6333")

    collection_name = "DemoImgEmbeddings"
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3, distance=Distance.EUCLID),
    )

    vectors = np.load("./embeddings/frames.npy")

    fd = open("./embeddings/names.json")
    # payload is now an iterator over file names
    payload = map(json.loads, fd)

    client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=payload,
        ids=None,  # Vector ids will be assigned automatically
        batch_size=256,  # How many vectors will be uploaded in a single request?
    )

    print(" - Uploaded " + str(vectors.shape[0]) + " vectors to QDrant")

if __name__ == "__main__":
    upload_to_qdrant()