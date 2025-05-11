from flask import Flask, request, jsonify
from predict import predict_news

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Check if batch prediction
        if isinstance(data, list):
            predictions = []
            for item in data:
                text = item.get('text', '')
                if not text:
                    predictions.append({'error': 'No text provided'})
                else:
                    predictions.append({'text': text, 'prediction': predict_news(text)})
            return jsonify(predictions), 200
        
        # Single prediction
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        prediction = predict_news(text)
        return jsonify({'text': text, 'prediction': prediction}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to Fake News Detector API!'}), 200

if __name__ == '__main__':
    app.run(debug=True)
