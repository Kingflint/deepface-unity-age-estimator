from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "DeepFace API is running!"})


@app.route("/analyze", methods=["POST"])
def analyze():
    return jsonify({"error": "not implemented"}), 501


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)