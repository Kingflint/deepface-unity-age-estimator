from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        req_data = request.get_json()
        encoded_image = req_data.get("image")

        if not encoded_image:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 to image
        image_data = base64.b64decode(encoded_image)
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Analyze using DeepFace
        result = DeepFace.analyze(img, actions=['emotion', 'age', 'gender'], enforce_detection=False)

        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
