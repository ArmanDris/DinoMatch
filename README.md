# DinoMatch

![DinoMatch demo](data/Demo.gif)

This image similarity search combines facebook's dino ViT Embedder and QDrant's vector similarity search. There is also a rudimentary RGB Embedder  that is useful for understanding how similarity search works in QDrant.

## Getting Started:

1. Pull & run the QDrant docker container ([more info here](https://qdrant.tech/documentation/quick-start/))
 - `$ docker pull qdrant/qdrant`
 - `$ docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`

2. Create & activate conda environment using environment.yml ([install conda here](https://docs.anaconda.com/free/miniconda/))
 - `$ conda env create --file environment.yml`
 - `$ conda activate torch`

3. Build the UI
 - `$ cd ui`
 - `$ npm install`
 - `$ npm run build`

4. Run the script
 - `$ python main.py`
    - This will use the vision transformer to create vector embeddings for all images in the weeds dataset
    - Then the vector embeddings and their names will be uploaded to the local QDrant service
    - Finally the flask server will start and you can go to [localhost:8010](http://localhost:8010) to try out DinoMatch!
    - (If you pass -t/--test to main.py then it will run all tests instead)

## Sources:

CropVsWeed dataset from:
https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes?resource=download

## Documentation:

VitImageEmbedder(src_folder)
Given a source folder this class will use facebook's dino-vits to encode each .jpg image as a vector. It will then connect to QDrant and upload each embedding along with some metadata about the image as a payload.

RgbImageEmbedder(src_folder, dst_file)
Given a source folder and destination file this class will encode each .png image as a numpy vector with its average rgb value [r, g ,b]. The results are stored in a dictionary with the file name being the key and the vector being the value

UploadDictToQdrant(src_file)
Will load a dictionary from src_file and upload the values in the dictionary as the vectors and the key as the 'file name' payload.

Using RgbImageEmbedder and UploadDictToQdrant together would look like:
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