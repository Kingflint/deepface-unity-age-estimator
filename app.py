from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from deepface import DeepFace
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "DeepFace API is running!"})

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        req_data = request.get_json()
        encoded_image = req_data.get("image")

        if not encoded_image:
            return jsonify({"error": "No image data provided"}), 400

        image_data = base64.b64decode(encoded_image)
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        result = DeepFace.analyze(img, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 🟢 Get dynamic port from Render
    app.run(host="0.0.0.0", port=port, debug=True)  # 🟢 Bind to 0.0.0.0
