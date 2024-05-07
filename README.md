# DemoImageEmbedder

Right now this is a proof of concept for using a Vector Database as "Memory" for Agrobot

## Usage:

Start QDrant service:
1. docker run -p 6333:6333 -p 6334:6334 \
    qdrant/qdrant

2. python3 run.py (-t)
    - This will encode the images as vectors and and a matching JSON file will be created with each vector's name
    - Then the vector embeddings and their names will be uploaded to QDrant
    - (-t, --test) will run tests

python3 -m unittest tests.test_embed

## Interesting Resources:
(Article Where the nature DeepWeeds repo comes from:
https://www.nature.com/articles/s41598-018-38343-3)

(Article of someone doing similarity search on dresses using image embeddings and qdrant:
https://medium.com/picsellia/how-we-built-a-dataset-visual-similarity-search-feature-by-using-embeddings-and-qdrant-ec9787383058)

(Another interesting article with number recognition:
https://docs.voxel51.com/tutorials/qdrant.html)

(Dandelion vs Radish Dataset: 
https://www.kaggle.com/datasets/junglepy/weeder-dandelions-vs-radishes/data)