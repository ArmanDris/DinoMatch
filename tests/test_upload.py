import unittest
import os
import pickle
import tempfile
import numpy as np
import upload_dict_to_qdrant
import rgb_image_embedder

class TestUploadDictToQdrant(unittest.TestCase):

    def setUp(self):
        self.temp_data_file = tempfile.NamedTemporaryFile(delete=False)
        rgb_image_embedder.RgbImageEmbedder(src_folder="data/10_Demo_Images", dst_file=self.temp_data_file.name)
        
        self.uploader = upload_dict_to_qdrant.UploadDictToQdrant(src_file=self.temp_data_file.name);

    def tearDown(self):
        self.temp_data_file.close()
        try:
            os.unlink(self.temp_data_file.name)
        except FileNotFoundError:
            pass

    def test_UploadDictToQdrant__load_dictionary(self):
        # names should include each of ["1.png", "2.png", ... "10.png"]
        # each name should be associate with the correct vector

        grn_v = np.array([0, 255, 0])
        red_v = np.array([255, 0, 0])

        points = {"1.png": grn_v, "2.png": grn_v, "3.png": grn_v, "4.png": red_v, "5.png": red_v, 
                  "6.png": grn_v, "7.png": red_v, "8.png": grn_v, "9.png": grn_v, "10.png": grn_v}
        
        for payload, vec in zip(self.uploader.names, self.uploader.nmpy_vectors):
            correct_vec = points.pop(payload["file_name"])
            np.testing.assert_array_equal(vec, correct_vec)
        

if __name__ == "__main__":
    unittest.main()