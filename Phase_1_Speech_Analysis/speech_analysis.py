import whisper
import librosa
import numpy as np
import re
import os
from moviepy.editor import VideoFileClip

# --- PHASE 1: ADVANCED SPEECH INTELLIGENCE ---
# "Responsible AI" Update: analyzing Prosody, not just words.

class AudioEngine:
    def __init__(self):
        print("⏳ Loading Whisper Model (Phase 1 Engine)...")
        # 'tiny' is fast; use 'base' if your server allows
        self.model = whisper.load_model("tiny")

    def extract_audio(self, video_path):
        """Extracts WAV audio from MP4 video safely."""
        audio_path = "temp_phase1.wav"
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
            video.close()
            return audio_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None

    def analyze_prosody(self, audio_path):
        """
        The 'Responsible Reasoning' Engine.
        Analyzes Volume (Confidence) and Pauses (Hesitation).
        """
        y, sr = librosa.load(audio_path)
        
        # 1. Volume / Energy Analysis (Root Mean Square)
        # Responsible Logic: Low volume often correlates with low confidence.
        rms = librosa.feature.rms(y=y)[0]
        avg_volume = np.mean(rms)
        # Normalize (0.05 is a typical average speech RMS)
        vol_score = min(max(avg_volume / 0.05, 0), 1) * 100 
        
        # 2. Pause / Silence Detection
        # Responsible Logic: High silence ratio + Low WPM = Nervousness/Forgetting.
        non_silent_intervals = librosa.effects.split(y, top_db=25)
        total_duration = librosa.get_duration(y=y, sr=sr)
        non_silent_duration = sum([(end - start) / sr for start, end in non_silent_intervals])
        
        pause_ratio = 1.0 - (non_silent_duration / total_duration) if total_duration > 0 else 0
        
        # 3. Advanced Pitch (Fundamental Frequency - F0)
        # Filter out noise to find true voice inflection
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        mask = magnitudes > np.median(magnitudes)
        active_pitches = pitches[mask]
        # Filter for human speech range (70Hz - 400Hz)
        valid_pitches = active_pitches[(active_pitches > 70) & (active_pitches < 400)]
        
        pitch_std = np.std(valid_pitches) if len(valid_pitches) > 0 else 0
        
        return {
            "volume_score": round(vol_score, 1),
            "pause_ratio": round(pause_ratio * 100, 1), # % of time spent silent
            "pitch_variation": round(float(pitch_std), 2)
        }

    def generate_viva_topics(self, transcript):
        """
        Extracts key technical terms for Faculty Questions.
        """
        stopwords = set(["the", "and", "is", "of", "to", "a", "in", "that", "it", "for", "on", "with", "as", "was", "at", "um", "uh", "like", "actually", "so", "you", "my", "we", "are"])
        # Find words longer than 3 chars
        words = re.findall(r'\b\w+\b', transcript.lower())
        
        word_counts = {}
        for w in words:
            if w not in stopwords and len(w) > 3:
                word_counts[w] = word_counts.get(w, 0) + 1
        
        # Return top 5 most frequent keywords
        topics = [w.title() for w, count in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:5]]
        return topics if topics else ["General Content"]

    def analyze(self, video_path):
        print(f"--- 🔍 Starting Phase 1 Analysis on: {video_path} ---")
        
        # 1. Extraction
        audio_path = self.extract_audio(video_path)
        if not audio_path: return

        # 2. Transcription (AI Listening)
        result = self.model.transcribe(audio_path)
        transcript = result["text"]
        
        # 3. Physics Analysis (Prosody)
        prosody = self.analyze_prosody(audio_path)
        
        # 4. Word Metrics
        words = transcript.split()
        duration = librosa.get_duration(filename=audio_path)
        wpm = round(len(words) / (duration / 60), 1) if duration > 0 else 0
        
        fillers = ["um", "uh", "like", "actually", "basically"]
        filler_count = sum(1 for w in words if re.sub(r'[^\w]', '', w.lower()) in fillers)

        # 5. Viva Topic Generation
        topics = self.generate_viva_topics(transcript)

        # 6. Cleanup
        if os.path.exists(audio_path): os.remove(audio_path)

        # --- OUTPUT REPORT ---
        print("\n📊 --- ANALYSIS RESULTS ---")
        print(f"📝 Transcript Snippet: \"{transcript[:100]}...\"")
        print(f"🗣️  Speaking Rate: {wpm} WPM")
        print(f"⏸️  Pause Ratio: {prosody['pause_ratio']}% (Silence/Hesitation)")
        print(f"🔊 Volume Confidence: {prosody['volume_score']}/100")
        print(f"🎼 Intonation (Pitch Std): {prosody['pitch_variation']} (Higher is better)")
        print(f"🤔 Fillers Detected: {filler_count}")
        print(f"🙋 Suggested Viva Topics: {', '.join(topics)}")
        
        print("\n✅ Phase 1 Analysis Complete.")

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Create a dummy instance to test logic if run directly
    engine = AudioEngine()
    print("System Ready. Call engine.analyze('your_video.mp4') to test.")
