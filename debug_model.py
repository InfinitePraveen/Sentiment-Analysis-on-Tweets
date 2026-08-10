import joblib
from utils.preprocessing import preprocess_tweet

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
text = "I love this app"
vec = vectorizer.transform([preprocess_tweet(text)])
print("classes:", getattr(model, "classes_", None))
print("predict:", model.predict(vec))
if hasattr(model, "predict_proba"):
    print("proba:", model.predict_proba(vec))
