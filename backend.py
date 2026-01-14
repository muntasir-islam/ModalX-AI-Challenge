import os

# Set FFMPEG paths explicitly for Linux environments
os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

import re
import cv2
import whisper
import yt_dlp
import librosa
import numpy as np
import mediapipe as mp
import gdown
import pytesseract
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from fpdf import FPDF
from datetime import datetime
from emotion_engine import EmotionAnalyzer

DEVICE = "cpu"
print(f"ModalX Backend initialized on: {DEVICE}")

# --- SCORING WEIGHTS ---
WEIGHTS = {
    "AUDIO": 0.30,   # 30%
    "VISUAL": 0.30,  # 30%
    "EMOTION": 0.20, # 20%
    "CONTENT": 0.20  # 20%
}

class VideoDownloader:
    @staticmethod
    def download_from_url(url, output_filename="input_video.mp4"):
        if "drive.google.com" in url:
            try:
                download_path = gdown.download(url, output_filename, quiet=False, fuzzy=True)
                return download_path if download_path else None
            except: return None
        
        ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': output_filename, 'quiet': True, 'overwrites': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return output_filename if os.path.exists(output_filename) else None
        except: return None

class SpeechAnalyzer:
    def __init__(self):
        self.whisper_model = whisper.load_model("base", device=DEVICE)
        # Dictionary for Content Impact Algorithm
        self.power_words = {
            "significant", "therefore", "consequently", "demonstrate", "specifically",
            "critical", "essential", "methodology", "conclusion", "result", "impact",
            "strategy", "implementation", "analysis", "evidence", "innovative"
        }
        self.transitional_phrases = {
            "on the other hand", "in addition", "furthermore", "however", "for example",
            "as a result", "in conclusion", "firstly", "secondly"
        }

    def analyze_content(self, transcript):
        """Evaluates the 'goodness' of sentence delivery and vocabulary."""
        words = transcript.lower().split()
        total_words = len(words)
        if total_words == 0: return 0, 0
        
        # 1. Vocabulary Impact Score
        impact_count = sum(1 for w in words if w in self.power_words)
        impact_score = min((impact_count / total_words) * 100 * 5, 100) # Scaling factor
        
        # 2. Flow/Structure Score (Transitional phrases)
        transcript_lower = transcript.lower()
        flow_count = sum(1 for phrase in self.transitional_phrases if phrase in transcript_lower)
        flow_score = min(flow_count * 10, 100) # 10 points per transition
        
        # Average
        final_content_score = (impact_score * 0.6) + (flow_score * 0.4)
        return round(final_content_score, 1), impact_count

    def analyze_prosody(self, audio_path):
        y, sr = librosa.load(audio_path)
        rms = librosa.feature.rms(y=y)[0]
        vol_score = min(max(np.mean(rms) / 0.05, 0), 1) * 100 
        
        non_silent = librosa.effects.split(y, top_db=25)
        total_dur = librosa.get_duration(y=y, sr=sr)
        active_dur = sum([(end - start) / sr for start, end in non_silent])
        pause_ratio = 1.0 - (active_dur / total_dur) if total_dur > 0 else 0
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        active_pitches = pitches[magnitudes > np.median(magnitudes)]
        valid = active_pitches[(active_pitches > 70) & (active_pitches < 400)]
        pitch_std = np.std(valid) if len(valid) > 0 else 0
        
        return {"volume_score": round(vol_score, 1), "pause_ratio": round(pause_ratio*100, 1), "pitch_variation": round(float(pitch_std), 2)}

    def process_audio(self, video_path):
        audio_path = "temp_audio.wav"
        try:
            video = VideoFileClip(video_path)
            duration = video.duration
            video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
            video.close()
            
            result = self.whisper_model.transcribe(audio_path)
            transcript = result["text"]
            words = transcript.split()
            wpm = round(len(words) / (duration / 60), 1) if duration > 0 else 0
            
            fillers = ["um", "uh", "like", "actually", "basically", "literally", "so"]
            filler_count = sum(1 for w in words if re.sub(r'[^\w]', '', w.lower()) in fillers)
            
            physics = self.analyze_prosody(audio_path)
            content_score, impact_count = self.analyze_content(transcript)
            
            if os.path.exists(audio_path): os.remove(audio_path)
            return {
                "transcript": transcript, "wpm": wpm, "filler_count": filler_count, 
                "physics": physics, "content_score": content_score, "impact_words": impact_count
            }
        except:
            if os.path.exists(audio_path): os.remove(audio_path)
            return None

class VisualAnalyzer:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5)

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        analyzed = 0; eye_contact = 0; posture = 0; face_detected_frames = 0
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_count % 15 == 0:
                analyzed += 1
                try:
                    results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.face_landmarks:
                        face_detected_frames += 1
                        nose = results.face_landmarks.landmark[1].x
                        if abs(nose - 0.5) < 0.1: eye_contact += 1 # Simplified center check
                    if results.pose_landmarks:
                        shldr_diff = abs(results.pose_landmarks.landmark[11].y - results.pose_landmarks.landmark[12].y)
                        if shldr_diff < 0.05: posture += 1
                except: pass
            frame_count += 1
        cap.release()
        
        if analyzed == 0: return None
        is_slide_mode = (face_detected_frames / analyzed) < 0.10
        return {
            "is_slide_mode": is_slide_mode,
            "eye_contact_score": round((eye_contact/analyzed)*100, 1),
            "posture_score": round((posture/analyzed)*100, 1)
        }

class SlideAnalyzer:
    def analyze_slides(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_text_density = []; slide_changes = 0; last_mean = None
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % 60 == 0: # Check every 2 seconds roughly
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray)
                word_count = len(text.split())
                if word_count > 5: total_text_density.append(word_count)
                
                curr_mean = np.mean(cv2.resize(gray, (32, 32)))
                if last_mean is not None and abs(curr_mean - last_mean) > 5: slide_changes += 1
                last_mean = curr_mean
            count += 1
        cap.release()
        
        avg_words = sum(total_text_density) / len(total_text_density) if total_text_density else 0
        return {"avg_words_per_slide": round(avg_words, 1), "slide_changes": slide_changes, "readability_score": max(100 - (avg_words * 0.5), 0)}

class ModalXSystem:
    def __init__(self):
        self.audio = SpeechAnalyzer()
        self.visual = VisualAnalyzer()
        self.slides = SlideAnalyzer()
        self.emotion_engine = EmotionAnalyzer() # Your integrated emotion engine

    def _calculate_final_score(self, audio, visual, emotion_summary, slides=None):
        feedback = []
        
        # 1. Audio Score (30%)
        aud_score = 100
        if audio['wpm'] < 110 or audio['wpm'] > 170: aud_score -= 10
        if audio['physics']['pause_ratio'] > 25: aud_score -= 10
        if audio['physics']['pitch_variation'] < 20: aud_score -= 10; feedback.append("Audio: Monotone voice detected.")
        aud_score = max(0, aud_score)

        # 2. Visual Score (30%)
        vis_score = 100
        if visual['is_slide_mode'] and slides:
            if slides['avg_words_per_slide'] > 50: vis_score -= 15; feedback.append("Visual: Slides are too text-heavy.")
            if slides['slide_changes'] < 3: vis_score -= 10
        else:
            if visual['eye_contact_score'] < 50: vis_score -= 15; feedback.append("Visual: Poor eye contact.")
            if visual['posture_score'] < 70: vis_score -= 10

        # 3. Emotion Score (20%)
        emo_score = 50 # Default
        if emotion_summary:
            pos = emotion_summary.get('happy', 0) + emotion_summary.get('neutral', 0) * 0.8 + emotion_summary.get('surprised', 0)
            neg = emotion_summary.get('sad', 0) + emotion_summary.get('fear', 0) + emotion_summary.get('angry', 0)
            total = sum(emotion_summary.values())
            if total > 0:
                emo_score = (pos / total) * 100
            if emo_score < 60: feedback.append("Emotion: Tone appears nervous or low energy.")
        
        # 4. Content Impact Score (20%)
        cont_score = audio['content_score']
        if cont_score > 70: feedback.append("Content: Excellent use of professional vocabulary.")
        elif cont_score < 40: feedback.append("Content: Sentence structure is basic. Use more transitional phrases.")

        # Weighted Final Calculation
        final_score = (aud_score * WEIGHTS["AUDIO"]) + \
                      (vis_score * WEIGHTS["VISUAL"]) + \
                      (emo_score * WEIGHTS["EMOTION"]) + \
                      (cont_score * WEIGHTS["CONTENT"])
                      
        return int(final_score), feedback, emo_score, cont_score

    def generate_pdf(self, metrics, score, feedback, s_name, s_id, e_times, e_emotions, e_score, c_score):
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_fill_color(24, 40, 72)
        pdf.rect(0, 0, 210, 40, 'F')
        pdf.set_font("Arial", "B", 22); pdf.set_text_color(255, 255, 255)
        pdf.set_y(15); pdf.cell(0, 10, "ModalX Detailed Report", ln=True, align="C")
        
        # Student Info
        pdf.set_text_color(0, 0, 0); pdf.set_y(50)
        pdf.set_font("Arial", "B", 12); pdf.cell(0, 10, f"Student: {s_name} | ID: {s_id}", ln=True)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
        
        # Final Score
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 15, f"Final Weighted Score: {score}/100", ln=True, align="C")
        pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(5)

        # Detailed Breakdown
        pdf.set_font("Arial", "B", 12); pdf.cell(0, 10, "1. Performance Breakdown (Weighted)", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(50, 8, f"Audio Clarity ({int(WEIGHTS['AUDIO']*100)}%)", 1); pdf.cell(0, 8, f"WPM: {metrics['audio']['wpm']} | Pitch Var: {metrics['audio']['physics']['pitch_variation']}", 1, 1)
        pdf.cell(50, 8, f"Visual Engagement ({int(WEIGHTS['VISUAL']*100)}%)", 1); pdf.cell(0, 8, f"Eye Contact: {metrics['visual']['eye_contact_score']}%", 1, 1)
        pdf.cell(50, 8, f"Emotional Intel ({int(WEIGHTS['EMOTION']*100)}%)", 1); pdf.cell(0, 8, f"Score: {int(e_score)}/100 (Based on facial/tone analysis)", 1, 1)
        pdf.cell(50, 8, f"Content Impact ({int(WEIGHTS['CONTENT']*100)}%)", 1); pdf.cell(0, 8, f"Score: {int(c_score)}/100 ({metrics['audio']['impact_words']} power words used)", 1, 1)
        
        # Emotional Graph
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12); pdf.cell(0, 10, "2. Emotional Timeline Analysis", ln=True)
        
        if e_times:
            plt.figure(figsize=(7, 3))
            unique_emotions = sorted(list(set(e_emotions)))
            y_vals = [unique_emotions.index(e) for e in e_emotions]
            plt.plot(e_times, y_vals, marker='o', linestyle='-', color='#4b6cb7', markersize=4)
            plt.yticks(range(len(unique_emotions)), unique_emotions)
            plt.title("Emotion Flow Over Time")
            plt.xlabel("Seconds"); plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
            plt.savefig("temp_graph.png")
            plt.close()
            pdf.image("temp_graph.png", x=10, w=180)
            os.remove("temp_graph.png")
        
        # Feedback
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12); pdf.cell(0, 10, "3. AI Recommendations", ln=True)
        pdf.set_font("Arial", "", 10)
        for fb in feedback: pdf.multi_cell(0, 7, f"- {fb}")

        # Output
        temp = f"temp_{s_id}.pdf"
        pdf.output(temp)
        with open(temp, "rb") as f: data = f.read()
        os.remove(temp)
        return data

    def analyze(self, path, name, sid, is_url=False):
        vid_path = path
        if is_url: vid_path = VideoDownloader.download_from_url(path)
        if not vid_path: return None

        # Run Engines
        aud_res = self.audio.process_audio(vid_path)
        vis_res = self.visual.analyze_video(vid_path)
        e_times, e_emotions, e_summary = self.emotion_engine.predict(vid_path) # Emotion Engine
        
        slide_res = None
        if vis_res and vis_res['is_slide_mode']:
            slide_res = self.slides.analyze_slides(vid_path)

        # Calculate Scores
        score, feedback, e_score, c_score = self._calculate_final_score(aud_res, vis_res, e_summary, slide_res)
        metrics = {"audio": aud_res, "visual": vis_res, "slides": slide_res}
        
        # Generate Report
        report = self.generate_pdf(metrics, score, feedback, name, sid, e_times, e_emotions, e_score, c_score)

        if is_url and os.path.exists(vid_path): os.remove(vid_path)
        
        return {
            "score": score, 
            "metrics": metrics, 
            "feedback": feedback, 
            "report": report,
            "emotion_data": {"times": e_times, "emotions": e_emotions, "summary": e_summary}
        }
