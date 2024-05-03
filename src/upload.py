import numpy as np
from typing import Dict, Iterator, List
import pickle
import json
from qdrant_client import QdrantClient # type: ignore
from qdrant_client.models import VectorParams, Distance # type: ignore


class ImportRgbEmbeddings:
    
    def __init__(self, src_file: str = "./embeddings/RGB_embeddings.pickle"):
        self.names: list[Dict[str, str]] = []
        self.list_of_nmpy_vectors: list[np.ndarray] = []

        for key, value, in self._load_dictionary(src_file).items():
            self.names.append({"file_name": key})
            self.list_of_nmpy_vectors.append(value)

    def _load_dictionary(self, src_file: str) -> Dict[str, np.ndarray]:
        dict_from_src = {}
        try:
            with open(src_file, 'rb') as f:
                dict_from_src = pickle.load(f)
        except OSError as e:
            print(f"Error opening saved RGB embeddings {src_file} - {e}")
            
        return dict_from_src

    def get_name_iterator(self) -> Iterator[Dict[str, str]]:
        return iter(self.names)
    
    def get_vector_iterator(self) -> Iterator[np.ndarray]:
        return iter(self.list_of_nmpy_vectors)
        

def upload_to_qdrant():
    client = QdrantClient("http://localhost:6333")

    collection_name = "DemoImgEmbeddings"
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3, distance=Distance.EUCLID),
    )

    embeddings = ImportRgbEmbeddings()

    client.upload_collection(
        collection_name=collection_name,
        vectors=embeddings.get_vector_iterator(),
        payload=embeddings.names,
    )

    print(f" - Uploaded {client.count(collection_name).count} vectors to QDrant")

if __name__ == "__main__":
    upload_to_qdrant()
