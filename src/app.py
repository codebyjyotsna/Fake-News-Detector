from flask import Flask, request, jsonify
from predict import predict_news

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    prediction = predict_news(text)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
