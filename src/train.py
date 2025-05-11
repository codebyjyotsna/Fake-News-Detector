import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text, vectorize_text

# Load dataset
data = pd.read_csv('data/fake_or_real_news.csv')
data['text'] = data['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Vectorize text
X_train_vec = vectorize_text(X_train)
X_test_vec = vectorize_text(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(model, 'models/fake_news_detector.pkl')
