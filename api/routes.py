from flask import Flask, Blueprint, send_from_directory, current_app, jsonify, request
import os, base64, random
from qdrant_client import QdrantClient

main_blueprint = Blueprint("main", __name__)


@main_blueprint.route("/")
@main_blueprint.route("/<path:path>")
def serve(path="index.html"):
    return send_from_directory(current_app.static_folder, path)


@main_blueprint.route("/getSample", methods=["GET"])
def getSample():
    sample_images = {}
    src_folder = os.path.join(
        current_app.root_path, "../data/CropVsWeedDataset/agri_data/data"
    )

    for filename in os.listdir(src_folder):
        if filename.endswith(".jpeg"):
            if random.randint(1, 10) != 3:
                continue
            with open(os.path.join(src_folder, filename), "rb") as image:
                encoded_string = base64.b64encode(image.read()).decode("utf-8")
                sample_images[filename] = encoded_string
        if len(sample_images) >= 4:
            break

    return jsonify(sample_images)


@main_blueprint.route("/getSimilar", methods=["POST"])
def getSimilar():
    json = request.get_json()
    filename = json["file_name"]
    print(filename)

    client = QdrantClient("http://localhost:6333")

    images_from_qdrant = client.recommend(
        collection_name="VitWeedEmbeddings",
        positive=[0],
        with_payload=True,
        limit=8,
    )

    for hit in images_from_qdrant:
        print(hit.payload, "score:", hit.score)

    similar_images = {}
    similar_images["image1"] = base64.b64encode(b"12345").decode("utf-8")
    similar_images["image2"] = base64.b64encode(b"67890").decode("utf-8")
    similar_images["image3"] = base64.b64encode(b"abvds").decode("utf-8")

    return jsonify(similar_images)
