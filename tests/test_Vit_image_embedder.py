import sys
if __name__ == "__main__":
    # Imports do not work when tests run directly
    print("Error: cannot run tests directly, use 'python main.py -t'")
    sys.exit(0)

import unittest
import tempfile
import os
import utils.ViT_image_embedder

class testVitImageEmbedder:

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test__vit_encoder(self):
        pass

    def test_embed_folder(self):
        pass

    def test_upload_to_qdrant(self):
        pass

if __name__ == "__main__":
    print("Error cannot run tests directly, use 'python main.py -t'")
