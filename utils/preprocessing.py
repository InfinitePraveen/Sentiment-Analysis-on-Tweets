import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data for both older and newer package versions
for resource, package in [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_tweet(text):
    """
    Preprocess tweet text for sentiment analysis

    Args:
        text (str): Raw tweet text

    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove mentions and hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove emojis and special characters
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove numbers (optional)
    text = re.sub(r"\d+", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and short tokens
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back to string
    processed_text = " ".join(tokens)

    return processed_text


def clean_tweet_for_display(text):
    """
    Clean tweet for display purposes (less aggressive than preprocessing)

    Args:
        text (str): Raw tweet text

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text, flags=re.MULTILINE)

    # Remove mentions
    text = re.sub(r"@\w+", "[MENTION]", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
