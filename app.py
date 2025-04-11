from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("distress_detection_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET"])
def index():
    return "Distress Detection API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Transform and predict
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0].max()

        return jsonify({
            "text": text,
            "is_distress": bool(prediction),
            "confidence": round(float(prob), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
