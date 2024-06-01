import argparse
import unittest
from utils.ViT_image_embedder import VitImageEmbedder
from api import create_app

images_folder = "./data/CropVsWeedDataset/agri_data/data"

def run_tests():
    loader = unittest.TestLoader()
    test_suite = loader.discover(start_dir="tests")
    runner = unittest.TextTestRunner()
    runner.run(test_suite)

def main():
    VitImageEmbedder(images_folder, "VitWeedEmbeddings")
    app = create_app()
    app.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main or run tests!")
    parser.add_argument('-t', '--test', action='store_true', help='Run all tests')
    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        main()