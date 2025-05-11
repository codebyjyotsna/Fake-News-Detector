import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # Remove URLs, HTML tags, special characters, and stopwords
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def vectorize_text(texts):
    vectorizer = TfidfVectorizer(max_features=5000)
    return vectorizer.fit_transform(texts)
