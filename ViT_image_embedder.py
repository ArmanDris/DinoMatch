import torch, os, pickle
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client import models
from transformers import ViTImageProcessor, ViTModel

class VitImageEmbedder:
    def __init__(self, src_folder: str):
        self.data = {}
        self.collection_name = "VitWeedEmbeddings"

        # Init ViT
        self.device = torch.device("cpu")
        self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
        self.model = ViTModel.from_pretrained('facebook/dino-vits16').to(self.device)

        self._embed_folder(src_folder)
        self.upload_to_qdrant()
    
    def upload_to_qdrant(self):
        client = QdrantClient("http://localhost:6333")

        if client.collection_exists(self.collection_name):
            client.delete_collection(self.collection_name)

        client.create_collection(
            collection_name=self.collection_name, 
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        
        i = 0
        for key, value in self.data.items():
            client.upsert( 
                collection_name=self.collection_name,
                points=[models.PointStruct(
                    id=i,
                    vector=value,
                    payload={"file_name":key, "vector":value.tolist()}
                )]
            )
            i+=1
        print(f" - uploaded {client.count(self.collection_name).count} embeddings to QDrant")

    def _embed_folder(self, src_folder):
        for filename in os.listdir(src_folder):
            if filename.endswith(".jpeg"):
                jpg_path = os.path.join(src_folder, filename)
                self.data[filename] = self._vit_encoder(jpg_path)

        print(f" - loaded {len(self.data)} image embeddings")

    def _vit_encoder(self, image_path:str) -> np.ndarray:
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        return outputs.squeeze(0)

# This script creates image embeddings for each image in the dataset and uploads them to QDrant a vector database.
# From there we can preform similarity search. Lets find images that are most similar to 