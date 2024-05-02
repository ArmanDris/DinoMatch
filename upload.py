import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient("http://localhost:6333")

client.recreate_collection(
    collection_name = "DemoImgEmbeddings",
    vectors_config=VectorParams(size=3, distance=Distance.Euclidean)
)

fd = open("./embeddings/frames.npy")

vectors = np.load("./data/startup_vectors.npy")

client.upload_collection(
    collection_name="DemoImgEmbeddings",
    vectors=vectors,
    ids=None,  # Vector ids will be assigned automatically
    batch_size=256,  # How many vectors will be uploaded in a single request?
)