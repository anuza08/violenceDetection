import clip
import torch
import yaml
import os
import cv2
import numpy as np
from PIL import Image

class Model:
    def __init__(self, settings_path: str = os.path.join(os.path.dirname(__file__), 'settings.yaml')):
        settings_path = os.path.abspath(settings_path)
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        with open(settings_path, "r") as file:
            self.settings = yaml.safe_load(file)

        self.device = self.settings['model-settings']['device']
        self.model_name = self.settings['model-settings']['model-name']
        self.threshold = self.settings['model-settings']['prediction-threshold']
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.labels = self.settings['label-settings']['labels']
        self.labels_ = [f"a photo of {label}" for label in self.labels]
        self.text_features = self.vectorize_text(self.labels_)
        self.default_label = self.settings['label-settings']['default-label']

    @torch.no_grad()
    def transform_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image).convert('RGB')
        tf_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        return tf_image

    @torch.no_grad()
    def vectorize_text(self, text: list):
        tokens = clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(tokens)
        return text_features

    @torch.no_grad()
    def predict_(self, text_features: torch.Tensor, image_features: torch.Tensor):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T
        values, indices = similarity[0].topk(1)
        return values, indices

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        tf_image = self.transform_image(image)
        image_features = self.model.encode_image(tf_image)
        values, indices = self.predict_(text_features=self.text_features, image_features=image_features)
        label_index = indices[0].cpu().item()
        label_text = self.default_label
        model_confidance = abs(values[0].cpu().item())
        if model_confidance >= self.threshold:
            label_text = self.labels[label_index]

        return {
            'label': label_text,
            'confidence': model_confidance
        }

    def process_video(self, input_video_path: str, output_path: str, batch_size: int = 16) -> str:
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS) + 0.5)
        
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        frames_batch = []
        frame_count = 0
        print("Starting video processing...")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read frame or video ended.")
                break

            # Collect frames in a batch
            frames_batch.append(frame)
            frame_count += 1

            # Process the batch when it's full
            if frame_count % batch_size == 0 or not cap.isOpened():
                print(f"Processing batch of {batch_size} frames...")
                # Apply model to the batch of frames
                frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_batch]
                predictions = [self.predict(frame) for frame in frames_rgb]

                # Annotate the frames
                for i, frame in enumerate(frames_batch):
                    prediction = predictions[i]
                    label = prediction['label']
                    conf = prediction['confidence']
                    print(f"Frame {frame_count - batch_size + i + 1}: Label: {label}, Confidence: {conf:.2f}")
                    frame_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
                    cv2.putText(frame_bgr, f"{label.title()} ({conf:.2f})", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    out.write(frame_bgr)

                # Clear the batch after processing
                frames_batch = []

        cap.release()
        out.release()

        print("Video processing completed.")
        return output_path