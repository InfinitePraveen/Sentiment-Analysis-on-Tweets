from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils.preprocessing import preprocess_tweet
import os
import json
from datetime import datetime
import traceback

app = Flask(__name__)

# Load model and vectorizer with better error handling
model_path = "models/sentiment_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"


def train_fallback_model():
    """Train a default model from the repository dataset if saved artifacts are missing or invalid."""
    dataset_path = os.path.join("data", "twitter_sentiment_sample.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training data not found at {dataset_path}")

    df = pd.read_csv(
        dataset_path,
        encoding="latin-1",
        header=None,
        names=["target", "ids", "date", "flag", "user", "text"],
    )
    df["sentiment_encoded"] = df["target"].map({0: 0, 2: 1, 4: 2})

    texts = df["text"].fillna("").astype(str).apply(preprocess_tweet)
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    y = df["sentiment_encoded"].astype(int).to_numpy()

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="liblinear",
        class_weight="balanced",
    )
    model.fit(X, y)
    return model, vectorizer


def load_model_artifacts():
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Model files not found. Checked {model_path} and {vectorizer_path}"
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    if not hasattr(model, "predict") or not hasattr(vectorizer, "transform"):
        raise TypeError("Saved model or vectorizer is invalid")

    test_tweet = "test"
    processed = preprocess_tweet(test_tweet)
    test_vector = vectorizer.transform([processed])
    prediction = model.predict(test_vector)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(test_vector)
        if probabilities.shape[1] not in (2, 3):
            raise ValueError(
                f"Unexpected number of prediction classes: {probabilities.shape[1]}"
            )

    if hasattr(model, "classes_") and len(model.classes_) > 0:
        valid_labels = {0, 1, 2}
        if not set(model.classes_).issubset(valid_labels):
            raise ValueError(f"Unexpected model labels: {model.classes_}")

    return model, vectorizer


print("🚀 Starting Flask application...")
print("📂 Checking for model files...")

model = None
vectorizer = None

try:
    model, vectorizer = load_model_artifacts()
    print("✅ Model and vectorizer loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Vectorizer type: {type(vectorizer)}")
except Exception as e:
    print(f"⚠️ Saved model unavailable or incompatible: {e}")
    try:
        os.makedirs("models", exist_ok=True)
        model, vectorizer = train_fallback_model()
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        print("✅ Fallback model trained and saved successfully")
    except Exception as fallback_error:
        print(f"❌ Error loading model: {fallback_error}")
        traceback.print_exc()
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
        # Check if model is loaded
        if model is None or vectorizer is None:
            return (
                jsonify(
                    {
                        "error": "Model not loaded. Please train the model first by running model_training.ipynb"
                    }
                ),
                500,
            )

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        tweet = data.get("tweet", "")

        if not tweet or not tweet.strip():
            return jsonify({"error": "No tweet provided"}), 400

        print(f"📝 Analyzing tweet: {tweet[:50]}...")

        # Preprocess tweet
        processed_tweet = preprocess_tweet(tweet)
        print(f"Processed: {processed_tweet[:50]}...")

        # Vectorize
        tweet_vector = vectorizer.transform([processed_tweet])
        print(f"Vector shape: {tweet_vector.shape}")

        # Predict
        prediction = model.predict(tweet_vector)[0]
        print(f"Prediction: {prediction}")

        # Get probabilities if available
        if hasattr(model, "predict_proba"):
            probability_array = model.predict_proba(tweet_vector)[0]
            class_values = [int(label) for label in model.classes_]
            probability_map = {
                int(label): float(prob)
                for label, prob in zip(model.classes_, probability_array)
            }

            if sorted(class_values) == [0, 2]:
                probabilities = [0.0, 0.0, 0.0]
                probabilities[0] = probability_map.get(0, 0.0)
                probabilities[2] = probability_map.get(2, 0.0)
            else:
                probabilities = [0.0, 0.0, 0.0]
                for label, prob in probability_map.items():
                    if label == 0:
                        probabilities[0] = prob
                    elif label == 1:
                        probabilities[1] = prob
                    elif label == 2:
                        probabilities[2] = prob
            print(f"Probabilities: {probabilities}")
        else:
            # For models without predict_proba (like LinearSVC)
            # Use a workaround or return default values
            probabilities = [0.0, 0.0, 0.0]
            if prediction == 0:
                probabilities[0] = 1.0
            elif prediction == 1:
                probabilities[1] = 1.0
            elif prediction == 2:
                probabilities[2] = 1.0

        # Map prediction to sentiment
        class_values = [int(label) for label in getattr(model, "classes_", [0, 1, 2])]
        if sorted(class_values) == [0, 2]:
            sentiment_map = {0: "negative", 2: "positive"}
        else:
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map.get(int(prediction), "unknown")
        confidence = float(max(probabilities) * 100)

        # Add to history
        history_entry = {
            "tweet": tweet,
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        predictions_history.append(history_entry)

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
        print(f"❌ Error in prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/history")
def get_history():
    return jsonify({"history": predictions_history})


@app.route("/health")
def health_check():
    """Health check endpoint to verify model is loaded"""
    if model is not None and vectorizer is not None:
        return jsonify(
            {
                "status": "healthy",
                "model_loaded": True,
                "model_type": str(type(model)),
                "predictions_count": len(predictions_history),
            }
        )
    else:
        return (
            jsonify(
                {
                    "status": "unhealthy",
                    "model_loaded": False,
                    "error": "Model or vectorizer not loaded",
                }
            ),
            500,
        )


if __name__ == "__main__":
    print("🌐 Starting Flask app on http://localhost:5000")
    app.run(debug=True, port=5000)
