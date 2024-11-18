from flask import Flask, render_template, request, jsonify
from model import Model
import cv2
import numpy as np

app = Flask(__name__)

# Initialize the model
model = Model()  # Ensure settings.yaml is configured correctly

@app.route('/')
def home():
    return render_template('index.html')  # Render the UI

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image as a numpy array
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Get prediction from the model
    prediction = model.predict(image)

    return jsonify(prediction)  # Return JSON response

if __name__ == '__main__':
    app.run(debug=True)
