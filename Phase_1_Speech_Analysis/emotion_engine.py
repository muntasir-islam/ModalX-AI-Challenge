import os
import numpy as np
import librosa
import tensorflow as tf
from collections import Counter

class EmotionAnalyzer:
    def __init__(self, model_path='modalx_emotion_model.h5'):
        self.model_path = model_path
        self.model = None
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad', 'surprise', 'surprised']
        self.sample_rate = 22050
        self.duration = 3
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            return
        try:
            self.model = tf.keras.models.load_model(self.model_path)
        except:
            pass

    def predict(self, audio_path):
        if self.model is None: return [], [], {}
        
        try:
            y_full, sr = librosa.load(audio_path, sr=self.sample_rate)
        except: return [], [], {}

        chunk_samples = self.duration * sr
        num_chunks = int(librosa.get_duration(y=y_full, sr=sr) / self.duration)
        
        times, emotions = [], []
        
        for i in range(num_chunks):
            start = int(i * chunk_samples)
            y_chunk = y_full[start : int(start + chunk_samples)]
            
            if len(y_chunk) < chunk_samples: 
                y_chunk = np.pad(y_chunk, (0, int(chunk_samples - len(y_chunk))))

            mfccs = np.mean(librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=40).T, axis=0)
            pred = self.model.predict(np.expand_dims(np.expand_dims(mfccs, 0), 2), verbose=0)
            
            times.append(i * self.duration)
            emotions.append(self.emotions[np.argmax(pred)])

        return times, emotions, dict(Counter(emotions))
