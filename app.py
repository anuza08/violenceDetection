import os
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from model import Model
import cv2
import numpy as np

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
PROCESSED_DIR = os.path.join(BASE_DIR, 'static', 'processed')

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

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

    # Save the annotated image temporarily
    annotated_image_path = os.path.join('static', 'processed', f'annotated_{file.filename}')
    cv2.putText(image, f"{prediction['label']} ({prediction['confidence']:.2f})",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(annotated_image_path, image)

    # Return prediction and URL to the processed image
    image_url = url_for('static', filename=f'processed/annotated_{file.filename}')
    return jsonify({'label': prediction['label'], 'confidence': prediction['confidence'], 'image_url': image_url}), 200


def process_video(self, input_video_path: str, output_path: str, batch_size: int = 16) -> str:
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    batch_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Collect batch
        batch_frames.append(frame)
        if len(batch_frames) == batch_size:
            self._process_batch(batch_frames, out)
            batch_frames = []

    # Process any remaining frames
    if batch_frames:
        self._process_batch(batch_frames, out)

    cap.release()
    out.release()
    return output_path


def _process_batch(self, frames, video_writer):
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    predictions = [self.predict(f) for f in frames_rgb]

    for i, frame in enumerate(frames):
        prediction = predictions[i]
        label = prediction['label']
        conf = prediction['confidence']
        annotated_frame = cv2.putText(frame, f"{label} ({conf:.2f})", 
                                      (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video_writer.write(annotated_frame)

@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    input_video_path = os.path.join(UPLOAD_DIR, file.filename)
    output_video_path = os.path.join(PROCESSED_DIR, f'processed_{file.filename}')
    
    # Save the uploaded video
    file.save(input_video_path)

    # Process the video
    try:
        processed_video_path = model.process_video(input_video_path, output_video_path)

        # Example: Get prediction details (you might need to modify this depending on your model's output)
        video_label = "Violence Detected"  # Replace with actual label logic
        confidence = 0.95  # Replace with actual confidence logic

        video_url = url_for('static', filename=f'processed/processed_{file.filename}')
        return jsonify({
            'label': video_label,
            'confidence': confidence,
            'processed_video_url': video_url
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)