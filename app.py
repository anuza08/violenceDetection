from flask import Flask, render_template, request, jsonify, send_file, url_for
from model import Model
import cv2
import os
import numpy as np
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 
model = Model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    prediction = model.predict(image)
    return jsonify(prediction)

@app.route('/predict_video', methods=['POST'])
def predict_video():
    print("Received a video upload request.") 

    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    file = request.files['video']
    print(f"Uploaded video: {file.filename}") 

    if file.filename == '':
        return jsonify({'error': 'No selected video'}), 400

    # Save the uploaded video temporarily
    temp_video_path = os.path.join('static', 'uploads', file.filename)
    print(f"Saving video to: {temp_video_path}")
    file.save(temp_video_path)

    # Output path for the processed video
    output_video_path = os.path.join('static', 'processed', f'processed_{file.filename}')
    print(f"Output path for processed video: {output_video_path}") 

    # Process video using the Model class
    processed_video_path = model.process_video(temp_video_path, output_video_path)

    print(f"Processed video saved at: {processed_video_path}")  # Debug statement

     # Return the video URL to the frontend for playback
    video_url = url_for('static', filename=f'processed/processed_{file.filename}')
    return jsonify({'processed_video_url': video_url}), 200

if __name__ == '__main__':
    app.run(debug=True)

