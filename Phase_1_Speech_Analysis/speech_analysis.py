import os
import librosa
import numpy as np
import whisper
from moviepy.editor import VideoFileClip

if os.name == 'posix':
    os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"

class Phase1_SpeechAnalyzer:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.power_words = {
            "significant", "therefore", "demonstrate", "specifically", "critical", 
            "essential", "methodology", "conclusion", "impact", "strategy",
            "analysis", "implementation", "innovative", "furthermore", "consequently"
        }

    def extract_audio(self, video_path):
        temp_audio = "temp_phase1_audio.wav"
        try:
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(temp_audio, verbose=False, logger=None)
            clip.close()
            return temp_audio
        except:
            return None

    def analyze(self, video_path):
        audio_file = self.extract_audio(video_path)
        if not audio_file:
            return None

        try:
            result = self.model.transcribe(audio_file)
            transcript = result["text"]
            
            words = transcript.split()
            word_count = len(words)
            duration = librosa.get_duration(filename=audio_file)
            
            wpm = 0
            if duration > 0:
                wpm = round(word_count / (duration / 60), 1)

            y, sr = librosa.load(audio_file)
            non_silent_intervals = librosa.effects.split(y, top_db=25)
            active_time = sum([end - start for start, end in non_silent_intervals]) / sr
            pause_ratio = round((1.0 - (active_time / duration)) * 100, 1)
            
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            active_pitches = pitches[magnitudes > np.median(magnitudes)]
            pitch_std = 0
            if len(active_pitches) > 0:
                pitch_std = round(np.std(active_pitches[active_pitches > 0]), 2)

            impact_count = sum(1 for w in words if w.lower().strip(".,!?") in self.power_words)
            content_score = min((impact_count / word_count) * 500, 100) if word_count > 0 else 0

            if os.path.exists(audio_file):
                os.remove(audio_file)

            return {
                "transcript_preview": transcript[:150] + "...",
                "metrics": {
                    "wpm": wpm,
                    "duration_sec": round(duration, 2),
                    "pitch_variation": pitch_std,
                    "pause_ratio": pause_ratio,
                    "impact_words_count": impact_count,
                    "content_score": round(content_score, 1)
                },
                "feedback": self._generate_feedback(wpm, pitch_std, pause_ratio, impact_count)
            }

        except:
            if os.path.exists(audio_file): os.remove(audio_file)
            return None

    def _generate_feedback(self, wpm, pitch, pause, impact):
        feedback = []
        if wpm < 110: feedback.append("Speed: Too slow. Aim for 130-150 WPM.")
        elif wpm > 170: feedback.append("Speed: Too fast. Slow down for clarity.")
        
        if pitch < 20: feedback.append("Tone: Voice is monotone. Vary your pitch.")
        
        if impact < 3: feedback.append("Content: Use more professional vocabulary.")
        
        return feedback

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="test_video.mp4")
    args = parser.parse_args()

    if os.path.exists(args.video):
        analyzer = Phase1_SpeechAnalyzer()
        results = analyzer.analyze(args.video)
        print(results)
