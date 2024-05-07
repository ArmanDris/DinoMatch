import argparse
import sys
import unittest
import embed
import upload

def run_tests():
    loader = unittest.TestLoader()
    test_suite = loader.discover(start_dir="tests")
    runner = unittest.TextTestRunner()
    runner.run(test_suite)

def main():
    src_folder = "./data/10_Demo_Images"
    embeddings_file = "./data/embeddings/RGB_embeddings.pickle"

    embed.RgbImageEmbedder(src_folder=src_folder, dst_file=embeddings_file)
    upload.UploadDictToQdrant(src_file=embeddings_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main or run tests!")
    parser.add_argument('-t', '--test', action='store_true', help='Run all tests')
    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        main()