from flask import Flask, request, jsonify
from deepface import DeepFace
from io import BytesIO
import base64
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ DeepFace Age Estimator is Live!"

@app.route("/predict-age", methods=["POST"])
def predict_age():
    try:
        data = request.json
        image_data = base64.b64decode(data["image"])
        img = Image.open(BytesIO(image_data)).convert("RGB")
        img_arr = np.array(img)

        result = DeepFace.analyze(img_arr, actions=["age"], enforce_detection=False)
        age = result[0]["age"]
        return jsonify({"status": "success", "age": int(age)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    print("✅ Flask app has started")
    app.run(host="0.0.0.0", port=5000)
