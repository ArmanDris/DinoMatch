# DemoImageEmbedder

Right now this is a proof of concept for using a Vector Database as "Memory" for Agrobot

## Usage:

Start QDrant service:
1. docker run -p 6333:6333 -p 6334:6334 \
    qdrant/qdrant

2. python main.py (-t)
    - This will encode the images as vectors and and a matching JSON file will be created with each vector's name
    - Then the vector embeddings and their names will be uploaded to QDrant
    - (-t, --test) will run tests

python3 -m unittest tests.test_embed

## Sources:

CropVsWeed dataset from:
https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes?resource=download

## Documentation:

RgbImageEmbedder(src_folder, dst_file)
Given a source folder and destination file this class will encode each .png image as a numpy vector with its average rgb value [r, g ,b]. The results are stored in a dictionary with the file name being the key and the vector being the value

UploadDictToQdrant(src_file)
Will load a dictionary from src_file and upload the values in the dictionary as the vectors and the key as the 'file name' payload.

So to use these two together you would:
    src_folder = "./data/10_Demo_Images"
    embeddings_file = "./data/embeddings/RGB_embeddings.pickle"

    rgb_image_embedder.RgbImageEmbedder(src_folder=src_folder, dst_file=embeddings_file)
    upload_dict_to_qdrant.UploadDictToQdrant(src_file=embeddings_file)

## Interesting Resources:
(Article Where the nature DeepWeeds repo comes from:
https://www.nature.com/articles/s41598-018-38343-3)

(Article of someone doing similarity search on dresses using image embeddings and qdrant:
https://medium.com/picsellia/how-we-built-a-dataset-visual-similarity-search-feature-by-using-embeddings-and-qdrant-ec9787383058)

(Another interesting article with number recognition:
https://docs.voxel51.com/tutorials/qdrant.html)

(Dandelion vs Radish Dataset: 
https://www.kaggle.com/datasets/junglepy/weeder-dandelions-vs-radishes/data)