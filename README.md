
# # 🚨 Violence Detection System

## 📝 Problem Statement
This project aims to develop a **Violence Detection System** that can accurately identify instances of violence in images and videos using a deep learning model. The system leverages the **CLIP (Contrastive Language-Image Pretraining)** model to predict whether a given visual input contains signs of violence. It processes both images and videos, annotating the predictions directly on the media.

---

https://github.com/user-attachments/assets/78e0b173-1cda-4a41-84c9-aaddcede972b



## 📂 Project Structure
```
|-- static
|   |-- uploads         # Stores uploaded images/videos
|   |-- processed       # Stores processed images/videos with predictions
|-- templates
|   |-- index.html      # Main HTML template for the UI
|-- app.py              # Main Flask application
|-- model.py            # Model implementation using CLIP
|-- settings.yaml       # Configuration file for model parameters
|-- requirements.txt    # Dependencies for the project
```

---

## 🌟 Features
✅ **Image Prediction:** Detects violence in uploaded images and returns the annotated image with prediction labels and confidence scores.

✅ **Video Prediction:** Processes video files in batches, annotating detected violence scenes directly on the frames.

✅ **Robust Model Integration:** Uses the powerful **CLIP** model for feature extraction and violence detection.

✅ **Batch Processing:** Efficiently handles video data in customizable batch sizes for optimized performance.

---

## ⚙️ Installation
1. **Clone the repository**
```bash
git clone <repository-link>
cd <project-directory>
```
2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate    # For Windows
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration
Update the **`settings.yaml`** file for model settings, device configuration, and labels.
```yaml
model-settings:
  device: "cuda"  # Use 'cuda' for GPU or 'cpu' for CPU
  model-name: "ViT-B/32"
  prediction-threshold: 0.5

label-settings:
  labels:
    - "violence"
    - "non-violence"
  default-label: "non-violence"
```

---

## 🚀 Usage
### Run the Application
```bash
python app.py
```
The app will be available at **`http://localhost:5000`**.

### Image Prediction
- Navigate to `/` and upload an image file.
- The system will predict the label and display the annotated image with a confidence score.

### Video Prediction
- Navigate to `/predict_video` and upload a video file.
- The system will annotate the video frames with detected violence instances and provide a link to download the processed video.

---

## 🔍 Code Overview
### `app.py`
- **`/` Route:** Renders the main index page for uploading images.
- **`/predict_image` Route:** Processes uploaded images, predicts labels, and returns an annotated image.
- **`/predict_video` Route:** Processes uploaded videos in batches, predicts labels for each frame, and returns the processed video.

### `model.py`
- The **`Model`** class initializes the CLIP model, preprocesses images, and generates predictions.
- The **`process_video`** method efficiently handles large video files by batching frames and applying model predictions in bulk.

---

## 📋 Example Output
- **Image Prediction**
```
Label: Violence
Confidence: 0.89
```
- **Video Prediction**
```
Frame 15: Label: Violence, Confidence: 0.92
Frame 30: Label: Non-Violence, Confidence: 0.75
```

---

## 📦 Dependencies
- **Flask**: Web framework to build the application.
- **OpenCV**: Used for image and video processing.
- **CLIP** (by OpenAI): Model for powerful feature extraction and prediction.
- **PyTorch**: Deep learning framework for model integration.
- **NumPy**: For numerical operations on data.
- **Pillow**: For image conversion.

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Enhancements
- Improve the model's accuracy with additional fine-tuning.
- Add real-time violence detection for live video feeds.
- Develop a more intuitive and interactive user interface.

---

## 👨‍💻 Author
**Anuja**

---

## 📜 License
This project is licensed under the **MIT License**.


<h2>Problem Statement</h2>
This project aims to develop a Violence Detection System that can accurately identify instances of violence in images and videos using a deep learning model. The system leverages the CLIP (Contrastive Language-Image Pretraining) model to predict whether a given visual input contains signs of violence. It processes both images and videos, annotating the predictions directly on the media.



https://github.com/user-attachments/assets/cc676574-a2bd-4e69-8967-d56f1976cd7d




<h2>Features</h2>

Image Prediction: Detects violence in uploaded images and returns the annotated image along with the prediction label and confidence score.

Video Prediction: Processes video files in batches, annotating detected violence scenes with confidence scores directly on the frames.

Robust Model Integration: Uses the powerful CLIP model for feature extraction and violence detection.

Batch Processing: Efficiently handles video data in customizable batch sizes for optimized performance.



<h2>Dependencies</h2>

Flask: Web framework to build the application.

OpenCV: Used for image and video processing.

CLIP (by OpenAI): Model for powerful feature extraction and prediction.

PyTorch: Deep learning framework for model integration.

NumPy: For numerical operations on data.

Pillow: For image conversion.
