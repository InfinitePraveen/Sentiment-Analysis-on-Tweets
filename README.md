# 🐦 Sentiment Analysis on Tweets

<p align="center">
  <b>Natural Language Processing • Machine Learning • Flask • Twitter Sentiment Classification</b>
</p>

<p align="center">
  <a href="https://github.com/InfinitePraveen/Sentiment-Analysis-on-Tweets">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?logo=github" alt="GitHub Repository">
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-Web%20App-green?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/NLP-Sentiment%20Analysis-orange" alt="NLP">
  <img src="https://img.shields.io/badge/License-Apache%202.0-red" alt="License">
</p>

---

## 📌 Overview

**Sentiment Analysis on Tweets** is an end-to-end Natural Language Processing (NLP) and Machine Learning project designed to analyze the emotional polarity of tweets.

The project takes raw tweet text, cleans and preprocesses it, transforms the text into numerical features using **TF-IDF**, and uses trained machine-learning models to classify the sentiment of the tweet.

The repository also includes a **Flask-based web application** that provides an interactive interface for entering tweets and receiving sentiment predictions.

The application is designed around a complete machine-learning workflow:

```text
Raw Tweet
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Machine Learning Model
    ↓
Sentiment Prediction
    ↓
Confidence / Probability
    ↓
Web Interface
```

---

## 🎯 Project Objectives

The main objectives of this project are:

* Perform sentiment analysis on Twitter data.
* Apply NLP techniques to noisy social-media text.
* Clean and normalize tweets before modeling.
* Convert textual data into numerical features using TF-IDF.
* Train and evaluate multiple machine-learning models.
* Compare different classification approaches.
* Save trained models for later inference.
* Build a lightweight Flask application for real-time predictions.
* Demonstrate an end-to-end NLP and ML deployment workflow.

---

## ✨ Key Features

* 🐦 Tweet sentiment classification
* 🧹 Automated tweet preprocessing
* 🔤 TF-IDF feature extraction
* 🤖 Multiple machine-learning models
* 📊 Model evaluation notebooks
* 💾 Persistent trained model artifacts
* 🌐 Flask web application
* ⚡ Real-time prediction API
* 📈 Prediction probability support
* 🔄 Automatic fallback model training
* 🧪 Model and API verification utilities
* 🎨 Frontend interface using HTML/CSS/JavaScript

The Flask application loads a saved model and TF-IDF vectorizer and can automatically train a fallback Logistic Regression model from the repository dataset if the saved artifacts are unavailable or incompatible.

---

## 🧠 Machine Learning Pipeline

### 1. Data Collection

The project uses a Twitter sentiment dataset stored in:

```text
data/twitter_sentiment_sample.csv
```

The application expects the dataset in the standard six-column format:

```text
target
ids
date
flag
user
text
```

The sentiment targets are mapped into three encoded classes:

```text
0 → Negative
1 → Neutral
2 → Positive
```

The repository contains the dataset under the `data/` directory.

---

### 2. Text Preprocessing

Tweets are noisy and contain many elements that can negatively affect traditional machine-learning models.

The project therefore uses a dedicated preprocessing module:

```text
utils/preprocessing.py
```

The preprocessing stage prepares tweets before they are passed to the vectorizer and classifier.

Typical preprocessing operations include:

* Converting text to a consistent case
* Removing unnecessary URLs
* Handling mentions
* Handling hashtags
* Removing unwanted characters
* Cleaning excessive whitespace
* Preparing text for feature extraction

---

### 3. TF-IDF Feature Extraction

After preprocessing, tweets are converted into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The fallback training pipeline uses:

```python
TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    sublinear_tf=True
)
```

This configuration allows the model to learn from both:

* Unigrams — individual words
* Bigrams — pairs of consecutive words

while limiting the feature space to the most useful terms.

---

## 🤖 Machine Learning Models

The repository contains trained model artifacts for multiple classification approaches:

```text
models/
├── sentiment_model.pkl
├── sentiment_model_nb.pkl
├── sentiment_model_svm.pkl
└── tfidf_vectorizer.pkl
```

These artifacts correspond to different sentiment-classification approaches and allow predictions to be performed without retraining the models every time.

### Logistic Regression

The primary fallback model is **Logistic Regression**.

Its configuration includes:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver="liblinear",
    class_weight="balanced"
)
```

The `class_weight="balanced"` configuration helps account for potential class imbalance in the training data.

### Naive Bayes

A trained Naive Bayes model is also included:

```text
sentiment_model_nb.pkl
```

Naive Bayes is a popular baseline for text classification because of its simplicity and efficiency.

### Support Vector Machine

The repository also contains an SVM-based sentiment model:

```text
sentiment_model_svm.pkl
```

SVMs are particularly useful for high-dimensional sparse text representations such as TF-IDF features.

---

## 📊 Model Evaluation

The repository includes dedicated notebooks for analyzing and evaluating the machine-learning workflow:

```text
notebooks/
├── data_exploration.ipynb
├── model_training.ipynb
└── model_evaluation.ipynb
```

### Data Exploration

`data_exploration.ipynb` is used to investigate the dataset and understand the underlying text and sentiment distribution.

### Model Training

`model_training.ipynb` contains the training workflow used to build the sentiment models.

### Model Evaluation

`model_evaluation.ipynb` is used to assess model performance and compare classification results.

The notebooks provide a reproducible environment for experimenting with the NLP pipeline.

---

# 🌐 Flask Web Application

The project includes a Flask-based web application through:

```text
app.py
```

The application provides an interface where users can submit tweet text and obtain a sentiment prediction.

### Application Flow

```text
User enters tweet
       ↓
Flask receives request
       ↓
Tweet preprocessing
       ↓
TF-IDF transformation
       ↓
Trained model prediction
       ↓
Sentiment + probability
       ↓
Result displayed to user
```

The application exposes the following primary routes:

```text
GET  /
POST /predict
```

The `/predict` endpoint accepts JSON data containing a tweet and returns the prediction result.

---

## 🔌 Prediction API

The prediction endpoint can be used with a JSON request.

### Request

```http
POST /predict
Content-Type: application/json
```

Example:

```json
{
  "tweet": "I really love this product!"
}
```

### Example Response

```json
{
  "sentiment": "Positive",
  "confidence": 0.91
}
```

> The exact response structure may depend on the current frontend/application implementation.

---

# 📁 Project Structure

```text
Sentiment-Analysis-on-Tweets/
│
├── data/
│   └── twitter_sentiment_sample.csv
│
├── models/
│   ├── sentiment_model.pkl
│   ├── sentiment_model_nb.pkl
│   ├── sentiment_model_svm.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── utils/
│   └── preprocessing.py
│
├── app.py
├── debug_model.py
├── verify_api.py
├── requirements.txt
├── LICENSE
└── README.md
```

The current repository contains these major directories and application files, including the data, model, notebook, frontend, utility, Flask, and verification components.

---

# 🛠️ Technologies Used

### Programming Language

* **Python**

### Machine Learning

* Scikit-learn
* Logistic Regression
* Naive Bayes
* Support Vector Machine
* TF-IDF

### Natural Language Processing

* NLTK
* Text preprocessing
* Token-based feature extraction

### Data Science

* NumPy
* Pandas
* Matplotlib
* Seaborn

### Web Development

* Flask
* HTML
* CSS
* JavaScript

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook

The repository's `requirements.txt` includes Flask, NLTK, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, Joblib and other supporting packages.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/InfinitePraveen/Sentiment-Analysis-on-Tweets.git
```

Move into the project directory:

```bash
cd Sentiment-Analysis-on-Tweets
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Once the dependencies are installed, start the Flask application:

```bash
python app.py
```

The application will start locally.

Open your browser and visit:

```text
http://127.0.0.1:5000
```

You can then enter a tweet and receive its predicted sentiment.

---

# 🧪 Running the Notebooks

If you want to reproduce the machine-learning workflow, open the notebooks directory:

```text
notebooks/
```

Recommended order:

### Step 1 — Explore the Dataset

```text
data_exploration.ipynb
```

### Step 2 — Train Models

```text
model_training.ipynb
```

### Step 3 — Evaluate Models

```text
model_evaluation.ipynb
```

This workflow allows you to understand the dataset, train the classifiers, and evaluate their performance.

---

# 🔄 Automatic Fallback Training

One useful feature of the Flask application is its fallback mechanism.

When the application starts, it first attempts to load:

```text
models/sentiment_model.pkl
models/tfidf_vectorizer.pkl
```

If the saved artifacts are unavailable or invalid, the application attempts to train a default Logistic Regression model using:

```text
data/twitter_sentiment_sample.csv
```

The resulting model and vectorizer are then saved back into the `models/` directory.

This makes the application more resilient when model artifacts are missing.

---

# 📈 Why TF-IDF?

TF-IDF is a classic and effective approach for converting text into numerical features.

It assigns greater importance to words that are:

* Frequent within a particular document
* Less frequent across the overall corpus

This makes TF-IDF particularly useful for traditional NLP classification problems.

Using both unigrams and bigrams allows the model to capture individual words as well as short phrases.

For example:

```text
"good"
```

and:

```text
"very good"
```

can become separate features.

---

# 💡 Example Predictions

| Tweet                           | Expected Sentiment |
| ------------------------------- | ------------------ |
| `I absolutely love this!`       | 😊 Positive        |
| `This is the best day ever.`    | 😊 Positive        |
| `I don't like this product.`    | 😞 Negative        |
| `This experience was terrible.` | 😞 Negative        |
| `The product arrived today.`    | 😐 Neutral         |

> These examples illustrate the intended classification task; actual predictions depend on the trained model.

---

# 🔬 Machine Learning Workflow

The complete project can be summarized as:

```text
                 ┌──────────────────┐
                 │ Twitter Dataset  │
                 └────────┬─────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Data Exploration   │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Text Preprocessing │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   TF-IDF Features  │
                └─────────┬──────────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ Logistic │ │  Naive   │ │   SVM    │
       │Regression│ │  Bayes   │ │          │
       └────┬─────┘ └────┬─────┘ └────┬─────┘
            │            │            │
            └────────────┼────────────┘
                         ▼
                ┌──────────────────┐
                │ Model Evaluation │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Flask Deployment │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Sentiment Result │
                └──────────────────┘
```

---

# 🧩 Troubleshooting

### Model Not Loading

If you encounter an error related to missing model files, verify that the following files exist:

```text
models/sentiment_model.pkl
models/tfidf_vectorizer.pkl
```

The application also contains a fallback training mechanism that can rebuild these artifacts from the included dataset.

### Dependency Problems

Create a fresh virtual environment and reinstall the dependencies:

```bash
python -m venv venv
```

Then activate the environment and run:

```bash
pip install -r requirements.txt
```

### Port Already in Use

If port `5000` is already being used by another application, stop that application or configure Flask to use another available port.

---

# 🚧 Future Improvements

Possible future enhancements include:

* [ ] Improve text preprocessing
* [ ] Experiment with advanced NLP models
* [ ] Add transformer-based models such as BERT
* [ ] Perform hyperparameter optimization
* [ ] Add cross-validation
* [ ] Add comprehensive model comparison
* [ ] Improve handling of emojis and slang
* [ ] Add multilingual sentiment analysis
* [ ] Add batch CSV prediction
* [ ] Add sentiment analytics dashboards
* [ ] Deploy the application to a cloud platform
* [ ] Add REST API documentation
* [ ] Add automated testing
* [ ] Add CI/CD using GitHub Actions
* [ ] Monitor model performance after deployment

---

# 🤝 Contributing

Contributions are welcome!

If you would like to improve this project:

1. Fork the repository.
2. Create a new branch.

```bash
git checkout -b feature/improvement
```

3. Make your changes.
4. Commit your changes.

```bash
git commit -m "Add new improvement"
```

5. Push the branch.

```bash
git push origin feature/improvement
```

6. Open a Pull Request.

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

* Natural Language Processing
* Sentiment Analysis
* Text Cleaning
* Feature Engineering
* TF-IDF
* Machine Learning Classification
* Logistic Regression
* Naive Bayes
* Support Vector Machines
* Model Evaluation
* Model Serialization
* Flask Application Development
* REST-style prediction endpoints
* Frontend and backend integration
* Machine Learning deployment concepts

---

# 👨‍💻 Author

**Praveen Kumar**

Data Science & Machine Learning Enthusiast

GitHub:
https://github.com/InfinitePraveen

---

# 📄 License

This project is licensed under the **Apache License 2.0**.

See the [`LICENSE`](LICENSE) file for the complete license text.

---

# ⭐ Support

If you found this project useful or educational, consider giving the repository a ⭐ on GitHub.

Your support helps encourage further development and open-source learning.

---

<p align="center">
  <b>🐦 Turning Tweets into Insights with NLP & Machine Learning 🤖</b>
</p>
