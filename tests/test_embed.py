import unittest
import os
import numpy as np
from embed import RgbImageEmbeddings, save_rgb_embeddings

class TestEmbed(unittest.TestCase):
    
    def test_RgbImageEmbeddings_add_image(self):
        src_folder = "./data/10_Demo_Images"
        collection = RgbImageEmbeddings()

        for filename in os.listdir(src_folder):
            if filename.endswith(".png"):
                collection.add_image(filename, os.path.join(src_folder, filename))

        green_embedding = np.array([0, 255, 0], dtype=np.uint8)
        red_embedding = np.array([255, 0, 0], dtype=np.uint8)

        self.assertTrue(np.array_equal(collection.data["1.png"], green_embedding))
        self.assertTrue(np.array_equal(collection.data["2.png"], green_embedding))
        self.assertTrue(np.array_equal(collection.data["3.png"], green_embedding))
        self.assertTrue(np.array_equal(collection.data["4.png"], red_embedding))
        self.assertTrue(np.array_equal(collection.data["5.png"], red_embedding))
        self.assertTrue(np.array_equal(collection.data["6.png"], green_embedding))
        self.assertTrue(np.array_equal(collection.data["7.png"], red_embedding))
        self.assertTrue(np.array_equal(collection.data["8.png"], green_embedding))
        self.assertTrue(np.array_equal(collection.data["9.png"], green_embedding))
        self.assertTrue(np.array_equal(collection.data["10.png"], green_embedding))

        # def test_RgbImageEmb

if __name__ == "__main__":
    unittest.main()