import unittest
from src.embed import RgbImageEmbeddings, save_rgb_embeddings

class TestEmbed(unittest.testCase):
    
    def test_RgbImageEmbeddings_add_image(self):
        src_folder = "./data/10_Demo_Images"
        collection = RgbImageEmbeddings()

        for filename in os.listdir():
            if filename.ends_with(".png"):
                collection.add_image(os.path.join(src_folder, filename))

        self.assertEqual(collection.self["1.png"], np.array([0, 255, 0]), dtype=np.uint8)
        self.assertEqual(collection.self["2.png"], np.array([0, 255, 0]), dtype=np.uint8)
        self.assertEqual(collection.self["3.png"], np.array([0, 255, 0]), dtype=np.uint8)
        self.assertEqual(collection.self["4.png"], np.array([255, 0, 0]), dtype=np.uint8)
        self.assertEqual(collection.self["5.png"], np.array([255, 0, 0]), dtype=np.uint8)
        self.assertEqual(collection.self["6.png"], np.array([0, 255, 0]), dtype=np.uint8)
        self.assertEqual(collection.self["7.png"], np.array([255, 0, 0]), dtype=np.uint8)
        self.assertEqual(collection.self["8.png"], np.array([0, 255, 0]), dtype=np.uint8)
        self.assertEqual(collection.self["9.png"], np.array([0, 255, 0]), dtype=np.uint8)
        self.assertEqual(collection.self["10.png"], np.array([0, 255, 0]), dtype=np.uint8)

if __name__ == "__main__":
    unittest.main()