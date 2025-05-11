import joblib
from preprocess import clean_text, vectorize_text

# Load model
model = joblib.load('models/fake_news_detector.pkl')

def predict_news(text):
    cleaned_text = clean_text(text)
    vectorized_text = vectorize_text([cleaned_text])
    prediction = model.predict(vectorized_text)
    return "Fake" if prediction[0] == 1 else "Real"

# Example usage
if __name__ == "__main__":
    sample_text = "Breaking news! Scientists discovered a new planet."
    print(predict_news(sample_text))
