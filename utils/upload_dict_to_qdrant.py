import numpy as np
from typing import Dict
import pickle
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


class UploadDictToQdrant:
    
    def __init__(self, src_file: str):
        self.names: list[Dict[str, str]] = []
        self.nmpy_vectors: list[np.ndarray] = []

        self._load_dictionary(src_file)
        
        client = QdrantClient("http://localhost:6333")
        self.collection_name = "DemoImgEmbeddings"

        if client.collection_exists(self.collection_name):
            client.delete_collection(self.collection_name)
        
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=3, distance=Distance.EUCLID),
        )

        client.upload_collection(
            collection_name=self.collection_name,
            vectors=iter(self.nmpy_vectors),
            payload=iter(self.names)
        )

        print(f" - Uploaded {client.count(self.collection_name).count} vectors to QDrant")

        client.close()

    def _load_dictionary(self, src_file: str) -> None:
        dict_from_src = {}

        try:
            with open(src_file, 'rb') as f:
                dict_from_src = pickle.load(f)
        except OSError as e:
            print(f"Error opening saved RGB embeddings {src_file} - {e}")
        
        for key, value, in dict_from_src.items():
            self.names.append({"file_name": key})
            self.nmpy_vectors.append(value)
