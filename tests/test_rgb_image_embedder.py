if __name__ == "__main__":
    # Imports do not work when tests run directly
    print("Error cannot run tests directly, use 'python main.py -t'")
    return 0

import unittest
import os
import pickle
import tempfile
import numpy as np
import utils.rgb_image_embedder as rgb_image_embedder

class TestRgbImageEmbedder(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_out_file = tempfile.NamedTemporaryFile(delete=False)
        self.collection = rgb_image_embedder.RgbImageEmbedder(src_folder="./data/10_Demo_Images", dst_file=self.temp_out_file.name)

    def tearDown(self) -> None:
        self.temp_out_file.close()
        try:
            os.unlink(self.temp_out_file.name)
        except FileNotFoundError:
            pass
    
    def test_RgbImageEmbeddings_embed_directory(self):

        collection = self.collection

        green_embedding = np.array([0, 255, 0], dtype=np.uint8)
        red_embedding = np.array([255, 0, 0], dtype=np.uint8)

        np.testing.assert_array_equal(collection.data["1.png"], green_embedding)
        np.testing.assert_array_equal(collection.data["2.png"], green_embedding)
        np.testing.assert_array_equal(collection.data["3.png"], green_embedding)
        np.testing.assert_array_equal(collection.data["4.png"], red_embedding)
        np.testing.assert_array_equal(collection.data["5.png"], red_embedding)
        np.testing.assert_array_equal(collection.data["6.png"], green_embedding)
        np.testing.assert_array_equal(collection.data["7.png"], red_embedding)
        np.testing.assert_array_equal(collection.data["8.png"], green_embedding)
        np.testing.assert_array_equal(collection.data["9.png"], green_embedding)
        np.testing.assert_array_equal(collection.data["10.png"], green_embedding)

    def test_RgbImageEmbeddings_save_as(self):

        with open(self.temp_out_file.name, 'rb') as file:
            loaded_data = pickle.load(file)
        
        self.assertTrue(self.collection.data, loaded_data)
