import sys
if __name__ == "__main__":
    # Imports do not work when tests run directly
    print("Error: cannot run tests directly, use 'python main.py -t'")
    sys.exit(0)

from PIL import Image
import unittest, torch
from transformers import ViTImageProcessor, ViTModel
from utils.ViT_image_embedder import VitImageEmbedder
from qdrant_client import QdrantClient

class testVitImageEmbedder(unittest.TestCase):

    def setUp(self):
        self.embedder = VitImageEmbedder.__new__(VitImageEmbedder)

        self.embedder.data = {}
        self.collection_name = "VitWeedEmbeddings"

        self.embedder.device = torch.device("cpu")
        self.embedder.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
        self.embedder.model = ViTModel.from_pretrained('facebook/dino-vits16').to(self.embedder.device)

    def test__vit_encoder(self):
        with Image.open("data/CropVsWeedDataset/agri_data/data/agri_0_3.jpeg") as img:
            img = img.convert("RGB")
            crop_image_embedding = self.embedder._vit_encoder(img)
            self.assertTrue(len(crop_image_embedding) == 384)

    def test_embed_folder(self):
        src_folder = "data/10_Demo_Images"
        num_embedded = self.embedder._embed_folder(src_folder)
        self.assertEqual(num_embedded, 10)

    def test_upload_to_qdrant(self):
        test_collection_name = "VitTestUpload"
        VitImageEmbedder("data/10_Demo_Images", test_collection_name)
        client = QdrantClient("http://localhost:6333")
        num_vectors = client.count(test_collection_name).count
        self.assertEqual(num_vectors, 10)
        client.delete_collection(test_collection_name)
