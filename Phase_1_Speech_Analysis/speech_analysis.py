import whisper
import librosa
import numpy as np
from moviepy.editor import VideoFileClip
import os

class AudioEngine:
    def __init__(self):
        print("Loading Whisper Model...")
        self.model = whisper.load_model("tiny")

    def extract_audio(self, video_path):
        audio_path = "temp_phase1.wav"
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, logger=None)
        return audio_path

    def analyze(self, video_path):
        audio_path = self.extract_audio(video_path)

        result = self.model.transcribe(audio_path)

        y, sr = librosa.load(audio_path)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_std = np.std(pitches[pitches > 0])

        print(f"--- Phase 1 Results ---")
        print(f"Transcript: {result['text'][:100]}...")
        print(f"Pitch Variation: {pitch_std:.2f}")

        os.remove(audio_path)

if __name__ == "__main__":
    print("Phase 1 Audio Engine Ready.")
