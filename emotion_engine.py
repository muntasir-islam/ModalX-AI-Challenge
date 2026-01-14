import os
import numpy as np
import librosa
import tensorflow as tf
from collections import Counter

class EmotionAnalyzer:
    def __init__(self, model_path='modalx_emotion_model.h5'):
        self.model_path = model_path
        self.model = None
        self.emotions = [
            'angry', 
            'disgust', 
            'fear', 
            'happy', 
            'neutral', 
            'sad', 
            'surprise', 
            'surprised'
        ]
        self.sample_rate = 22050
        self.duration = 3
        
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model Loaded Successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self, audio_path):
        if self.model is None:
            return None, None, {"error": "Model not loaded"}

        try:
            y_full, sr = librosa.load(audio_path, sr=self.sample_rate)
        except Exception as e:
            return None, None, {"error": str(e)}

        total_duration = librosa.get_duration(y=y_full, sr=sr)
        chunk_samples = self.duration * sr
        num_chunks = int(total_duration / self.duration)
        
        timeline_seconds = []
        detected_emotions = []

        for i in range(num_chunks):
            start = int(i * chunk_samples)
            end = int(start + chunk_samples)
            y_chunk = y_full[start:end]

            if len(y_chunk) < chunk_samples:
                y_chunk = np.pad(y_chunk, (0, int(chunk_samples - len(y_chunk))))

            mfccs = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)

            input_data = np.expand_dims(mfccs_mean, axis=0)
            input_data = np.expand_dims(input_data, axis=2)

            prediction = self.model.predict(input_data, verbose=0)
            predicted_index = np.argmax(prediction)
            
            emotion = self.emotions[predicted_index]
            timeline_seconds.append(i * self.duration)
            detected_emotions.append(emotion)

        summary = dict(Counter(detected_emotions))
        
        return timeline_seconds, detected_emotions, summary
