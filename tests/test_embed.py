import unittest
import os
import pickle
import tempfile
import numpy as np
import embed

class TestEmbed(unittest.TestCase):

    def setUp(self) -> None:
        self.collection = embed.RgbImageEmbeddings()
        self.collection.embed_directory(src_folder="./data/10_Demo_Images")
    
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
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_filename = temp.name

        try:
            with open(temp_filename, 'wb') as file:
                pickle.dump(self.collection.data, file)
            
            with open(temp_filename, 'rb') as file:
                loaded_data = pickle.load(file)
            
            self.assertTrue(self.collection.data, loaded_data)
        
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

if __name__ == "__main__":
    unittest.main()