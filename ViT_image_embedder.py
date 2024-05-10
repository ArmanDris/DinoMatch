import torch
import numpy as np
import os
import pickle
from qdrant_client import QdrantClient
from qdrant_client import models

class VitImageEmbedder:
    def __init__(self, src_folder: str):
        self.data = {}
        self._embed_folder(src_folder)
        self.upload_to_qdrant()
    
    def upload_to_qdrant(self):
        client = QdrantClient("http://localhost:6333")
        collection_name = "VitWeedEmbeddings"

        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name, 
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE)
        )
        
        i = 0
        for key, value in self.data.items():
            client.upsert( 
                collection_name=collection_name,
                points=[models.PointStruct(
                    id=i,
                    vector=value,
                    payload={"file_name":key, "vector":value.tolist()}
                )]
            )
            i+=1

    def _embed_folder(self, src_folder):
        for filename in os.listdir(src_folder):
            if filename.endswith(".jpeg"):
                jpg_path = os.path.join(src_folder, filename)
                self.data[filename] = self._vit_encoder(jpg_path)

    def _vit_encoder(self, image_path:str) -> np.ndarray:
        return np.array([0])

    def _save_as(self, dst_file):
        try:
            with open(dst_file, "wb") as f:
                pickle.dump(self.data, f)
        except OSError as e:
            print(f"Error could not save {dst_file} with VitEmbeddings - {e}")