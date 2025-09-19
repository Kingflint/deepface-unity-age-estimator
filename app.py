from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "DeepFace API is running!"})


@app.route("/analyze", methods=["POST"])
def analyze():
    req_data = request.get_json()
    encoded_image = req_data.get("image") if req_data else None
    if not encoded_image:
        return jsonify({"error": "No image data provided"}), 400
    return jsonify({"received": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)