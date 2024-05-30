import unittest
import os
import tempfile
import numpy as np
import utils.upload_dict_to_qdrant as upload_dict_to_qdrant
import utils.rgb_image_embedder as rgb_image_embedder
from qdrant_client import QdrantClient

class TestUploadDictToQdrant(unittest.TestCase):

    def setUp(self):
        self.temp_data_file = tempfile.NamedTemporaryFile(delete=False)
        rgb_image_embedder.RgbImageEmbedder(src_folder="data/10_Demo_Images", dst_file=self.temp_data_file.name)
        
        self.uploader = upload_dict_to_qdrant.UploadDictToQdrant(src_file=self.temp_data_file.name);

        grn_v = np.array([0, 255, 0])
        red_v = np.array([255, 0, 0])
        
        self.correct_points = {"1.png": grn_v, "2.png": grn_v, "3.png": grn_v, "4.png": red_v, "5.png": red_v, 
                               "6.png": grn_v, "7.png": red_v, "8.png": grn_v, "9.png": grn_v, "10.png": grn_v}

    def tearDown(self):
        self.temp_data_file.close()
        try:
            os.unlink(self.temp_data_file.name)
        except FileNotFoundError:
            pass

    def test_UploadDictToQdrant__load_dictionary(self):
        # names should include each of ["1.png", "2.png", ... "10.png"]
        # each name should be associate with the correct vector

        for payload, vec in zip(self.uploader.names, self.uploader.nmpy_vectors):
            correct_vec = self.correct_points.pop(payload["file_name"])
            np.testing.assert_array_equal(vec, correct_vec)

        # All dictionary elements should be popped
        self.assertFalse(self.correct_points)
        
    def test_UploadDictToQdrant_constructor(self):
        # Here we want to make sure qdrant has the vectors
        client = QdrantClient("http://localhost:6333")

        records, offset = client.scroll(self.uploader.collection_name, limit=10,  with_vectors=True)
        self.assertIsNone(offset)
        for record in records:
            correct_vec = self.correct_points.pop(record.payload["file_name"])
            np.testing.assert_array_equal(record.vector, correct_vec)
            print(f"Popped {record.payload['file_name']}, {record.vector}, {correct_vec}")

        # All dictionary elements should be popped
        self.assertFalse(self.correct_points)
            
if __name__ == "__main__":
    print("Error cannot run tests directy, use 'python main.py -t'")
