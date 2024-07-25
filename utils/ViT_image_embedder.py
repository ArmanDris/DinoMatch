import torch, os
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client import models
from transformers import ViTImageProcessor, ViTModel, logging as hf_logging
hf_logging.set_verbosity_error()

class VitImageEmbedder:

    def __init__(self, src_folder: str, collection_name: str):
        self.data = {}
        self.collection_name = collection_name

        # Init ViT
        self.device = torch.device("cpu")
        self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
        self.model = ViTModel.from_pretrained('facebook/dino-vits16').to(self.device)

        self._embed_folder(src_folder)
        self._upload_to_qdrant()

    def _upload_to_qdrant(self):
        client = QdrantClient("http://qdrant:6333")

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
                    payload={"file_name":key}
                )]
            )
            i+=1
        print(f" - uploaded {client.count(self.collection_name).count} embeddings to QDrant")

    def _embed_folder(self, src_folder) -> int:
        for filename in os.listdir(src_folder):
            if filename.endswith(".jpeg") or filename.endswith(".png"):
                with Image.open(os.path.join(src_folder, filename)) as img:
                    img = img.convert("RGB")
                    self.data[filename] = self._vit_encoder(img)

        print(f" - loaded {len(self.data)} image embeddings")
        return len(self.data)

    def _vit_encoder(self, image:Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        return outputs.squeeze(0)
