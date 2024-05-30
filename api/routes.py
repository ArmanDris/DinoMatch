from flask import Flask, Blueprint, send_from_directory, current_app, jsonify, request
import os, base64, random
from qdrant_client import QdrantClient, models

main_blueprint = Blueprint("main", __name__)

@main_blueprint.route("/")
@main_blueprint.route("/<path:path>")
def serve(path="index.html"):
    return send_from_directory(current_app.static_folder, path)


@main_blueprint.route("/getSample", methods=["GET"])
def getSample():

    src_folder = os.path.join(current_app.root_path, "../data/CropVsWeedDataset/agri_data/data")

    sample_images = {}

    for filename in os.listdir(src_folder):
        if filename.endswith(".jpeg"):
            if random.randint(1, 100) != 3:
                continue
            with open(os.path.join(src_folder, filename), "rb") as image:
                encoded_string = base64.b64encode(image.read()).decode("utf-8")
                sample_images[filename] = encoded_string
        if len(sample_images) >= 4:
            break

    return jsonify(sample_images)


@main_blueprint.route("/getSimilar", methods=["POST"])
def getSimilar():
    # Find the ID of the image we want to perform similarity search on
    json = request.get_json()
    filename = json["file_name"]

    client = QdrantClient("http://localhost:6333")

    point_tuple = client.scroll(
        collection_name="VitWeedEmbeddings",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="file_name", match=models.MatchValue(value=filename)
                ),
            ]
        ),
        limit=2,
        with_payload=True,
        with_vectors=False,
    )

    point = point_tuple[0][0]

    # Use point.id to get similar images
    similar_points = client.recommend(
        collection_name="VitWeedEmbeddings",
        positive=[point.id],
        with_payload=True,
        limit=8,
    )

    # Add similar images & score to dict
    src_folder = os.path.join(current_app.root_path, "../data/CropVsWeedDataset/agri_data/data")

    similar_images = {}

    for point in similar_points:
        with open(os.path.join(src_folder, point.payload['file_name']), "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode("utf-8")
            similar_images[point.payload['file_name']] = [point.score, encoded_string]

    # Finally return the dict to front end :D
    return jsonify(similar_images)
