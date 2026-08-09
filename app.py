from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from utils.preprocessing import preprocess_tweet
import os
import json
from datetime import datetime

app = Flask(__name__)

# Load model and vectorizer
model_path = "models/sentiment_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    print("Model not found. Please train the model first.")
    model = None
    vectorizer = None

# Store recent predictions
predictions_history = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        tweet = data.get("tweet", "")

        if not tweet:
            return jsonify({"error": "No tweet provided"}), 400

        # Preprocess tweet
        processed_tweet = preprocess_tweet(tweet)

        # Vectorize
        tweet_vector = vectorizer.transform([processed_tweet])

        # Predict
        prediction = model.predict(tweet_vector)[0]
        probabilities = model.predict_proba(tweet_vector)[0]

        # Map prediction to sentiment
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map.get(prediction, "unknown")
        confidence = float(max(probabilities) * 100)

        # Add to history
        predictions_history.append(
            {
                "tweet": tweet,
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        # Keep only last 20 predictions
        if len(predictions_history) > 20:
            predictions_history.pop(0)

        return jsonify(
            {
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": {
                    "negative": round(probabilities[0] * 100, 2),
                    "neutral": round(probabilities[1] * 100, 2),
                    "positive": round(probabilities[2] * 100, 2),
                },
                "history": predictions_history[-5:],  # Return last 5 predictions
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history")
def get_history():
    return jsonify({"history": predictions_history})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
