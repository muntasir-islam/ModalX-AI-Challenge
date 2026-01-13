import os

os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

import re
import cv2
import torch
import whisper
import yt_dlp
import librosa
import numpy as np
import mediapipe as mp
import gdown
import pytesseract
from moviepy.editor import VideoFileClip
from transformers import pipeline
from fpdf import FPDF
from datetime import datetime

DEVICE = "cpu"
print(f"ModalX Backend initialized on: {DEVICE}")

class VideoDownloader:
    @staticmethod
    def download_from_url(url, output_filename="input_video.mp4"):
        if "drive.google.com" in url:
            try:
                download_path = gdown.download(url, output_filename, quiet=False, fuzzy=True)
                return download_path if download_path else None
            except Exception as e:
                return None
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_filename,
            'quiet': True,
            'overwrites': True,
            'nocheckcertificate': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return output_filename if os.path.exists(output_filename) else None
        except Exception as e:
            return None

class SpeechAnalyzer:
    def __init__(self):
        self.whisper_model = whisper.load_model("tiny", device=DEVICE)

    def analyze_prosody(self, audio_path):
        y, sr = librosa.load(audio_path)
        rms = librosa.feature.rms(y=y)[0]
        vol_score = min(max(np.mean(rms) / 0.05, 0), 1) * 100 
        
        non_silent = librosa.effects.split(y, top_db=25)
        total_dur = librosa.get_duration(y=y, sr=sr)
        active_dur = sum([(end - start) / sr for start, end in non_silent])
        pause_ratio = 1.0 - (active_dur / total_dur) if total_dur > 0 else 0
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        mask = magnitudes > np.median(magnitudes)
        active_pitches = pitches[mask]
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
        except: return None

        try:
            result = self.whisper_model.transcribe(audio_path)
            transcript = result["text"]
            words = transcript.split()
            wpm = round(len(words) / (duration / 60), 1) if duration > 0 else 0
            
            fillers = ["um", "uh", "like", "actually", "basically", "literally", "so"]
            filler_count = sum(1 for w in words if re.sub(r'[^\w]', '', w.lower()) in fillers)
            
            physics = self.analyze_prosody(audio_path)
            
            if os.path.exists(audio_path): os.remove(audio_path)
            return {"transcript": transcript, "wpm": wpm, "filler_count": filler_count, "physics": physics}
        except:
            if os.path.exists(audio_path): os.remove(audio_path)
            return None

class VisualAnalyzer:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5)

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        analyzed = 0
        eye_contact = 0
        posture = 0
        face_detected_frames = 0
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % 15 == 0:
                analyzed += 1
                try:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.holistic.process(img)
                    
                    if results.face_landmarks:
                        face_detected_frames += 1
                        nose = results.face_landmarks.landmark[1].x
                        cheeks = (results.face_landmarks.landmark[234].x + results.face_landmarks.landmark[454].x) / 2
                        if abs(nose - cheeks) < 0.05: eye_contact += 1

                    if results.pose_landmarks:
                        shldr_diff = abs(results.pose_landmarks.landmark[11].y - results.pose_landmarks.landmark[12].y)
                        if shldr_diff < 0.05: posture += 1
                except: pass
            frame_count += 1
        cap.release()
        
        if analyzed == 0: return None

        face_ratio = face_detected_frames / analyzed
        is_slide_mode = face_ratio < 0.10

        return {
            "is_slide_mode": is_slide_mode,
            "eye_contact_score": round((eye_contact/analyzed)*100, 1) if not is_slide_mode else 0,
            "posture_score": round((posture/analyzed)*100, 1) if not is_slide_mode else 0
        }

class SlideAnalyzer:
    def analyze_slides(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_text_density = []
        slide_changes = 0
        last_frame_hash = None
        
        frame_interval = int(fps * 2)
        count = 0
        
        print("Scanning Slides for Text...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if count % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_small = cv2.resize(gray, (32, 32))
                
                text = pytesseract.image_to_string(gray)
                word_count = len(text.split())
                if word_count > 5:
                    total_text_density.append(word_count)

                curr_mean = np.mean(gray_small)
                if last_frame_hash is not None:
                    if abs(curr_mean - last_frame_hash) > 5:
                        slide_changes += 1
                last_frame_hash = curr_mean

            count += 1
        cap.release()

        avg_words = sum(total_text_density) / len(total_text_density) if total_text_density else 0
        
        return {
            "avg_words_per_slide": round(avg_words, 1),
            "slide_changes": slide_changes,
            "readability_score": max(100 - (avg_words * 0.5), 0)
        }

class ModalXSystem:
    def __init__(self):
        self.audio = SpeechAnalyzer()
        self.visual = VisualAnalyzer()
        self.slides = SlideAnalyzer()

    def calculate_score(self, audio, visual, slides=None):
        score = 100
        feedback = []
        
        # Audio Scoring
        if audio['wpm'] < 110: score -= 5; feedback.append("Audio: Speaking too slowly.")
        elif audio['wpm'] > 170: score -= 5; feedback.append("Audio: Speaking too fast.")
        
        if audio['physics']['pause_ratio'] > 30: 
            score -= 10
            feedback.append("Audio: Frequent hesitations detected.")

        if audio['physics']['pitch_variation'] < 15:
            score -= 5
            feedback.append("Audio: Voice is monotone.")

        # Visual/Slide Scoring
        if visual['is_slide_mode'] and slides:
            feedback.append("Mode: Slide Presentation Detected.")
            if slides['avg_words_per_slide'] > 60:
                score -= 15
                feedback.append(f"Visual: Slides are cluttered ({slides['avg_words_per_slide']} words/slide).")
            elif slides['avg_words_per_slide'] < 10:
                score -= 5
                feedback.append("Visual: Slides seem empty.")
            else:
                feedback.append("Visual: Good text density.")
            
            if slides['slide_changes'] < 3:
                score -= 10
                feedback.append("Visual: Low slide variety.")
        else:
            feedback.append("Mode: Presenter Detected.")
            if visual['eye_contact_score'] < 60: score -= 10; feedback.append("Body: Low eye contact.")
            if visual['posture_score'] < 80: score -= 5; feedback.append("Body: Unstable posture.")

        return max(0, score), feedback

    def generate_pdf(self, metrics, score, feedback, s_name, s_id):
        pdf = FPDF()
        pdf.add_page()
        
        # --- 1. HEADER DESIGN ---
        # Dark Blue Background
        pdf.set_fill_color(24, 40, 72)
        pdf.rect(0, 0, 210, 40, 'F')
        
        pdf.set_font("Arial", "B", 22)
        pdf.set_text_color(255, 255, 255)
        pdf.set_y(15)
        pdf.cell(0, 10, "ModalX AI Assessment Report", ln=True, align="C")
        
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, "Automated Multi-Modal Presentation Analysis", ln=True, align="C")
        
        # --- 2. STUDENT INFO BOX ---
        pdf.ln(20)
        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(240, 240, 240) # Light Gray
        pdf.rect(10, 50, 190, 25, 'F')
        
        pdf.set_y(55)
        pdf.set_font("Arial", "B", 12)
        pdf.set_x(15)
        pdf.cell(35, 8, "Student Name:", 0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(60, 8, s_name, 0)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(25, 8, "Date:", 0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(40, 8, datetime.now().strftime('%Y-%m-%d'), 0, 1)
        
        pdf.set_x(15)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(35, 8, "Student ID:", 0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(60, 8, s_id, 0)

        # --- 3. GRADE & SCORE ---
        letter_grade = "F"
        color = (220, 53, 69) # Red
        if score >= 80: letter_grade, color = "A+", (25, 135, 84) # Green
        elif score >= 70: letter_grade, color = "A", (13, 202, 240) # Cyan
        elif score >= 60: letter_grade, color = "B", (255, 193, 7) # Yellow
        
        pdf.ln(20)
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Executive Summary", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Score Box
        pdf.set_fill_color(color[0], color[1], color[2])
        pdf.rect(10, pdf.get_y(), 190, 25, 'F')
        
        pdf.set_y(pdf.get_y() + 5)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 18)
        pdf.cell(95, 15, f"Final Score: {score}/100", 0, 0, 'C')
        pdf.cell(95, 15, f"Grade: {letter_grade}", 0, 1, 'C')
        pdf.set_text_color(0, 0, 0)

        # --- 4. DETAILED METRICS WITH BARS ---
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Detailed Performance Metrics", ln=True)
        pdf.set_font("Arial", "", 10)
        
        # Helper to draw bars
        def draw_metric(label, value_str, percent):
            pdf.ln(6)
            pdf.cell(50, 6, label, 0)
            pdf.cell(30, 6, value_str, 0)
            
            # Background Bar (Gray)
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.set_fill_color(230, 230, 230)
            pdf.rect(x, y+1, 80, 4, 'F')
            
            # Foreground Bar (Blue)
            pdf.set_fill_color(70, 130, 180)
            width = min(max(percent, 0), 100) * 0.8 # Scale to 80mm
            if width > 0:
                pdf.rect(x, y+1, width, 4, 'F')
            pdf.ln(2)

        # Audio Metrics
        draw_metric("Speaking Pace", f"{metrics['audio']['wpm']} WPM", min(metrics['audio']['wpm']/150*100, 100))
        draw_metric("Pitch Variation", f"{metrics['audio']['physics']['pitch_variation']}", min(metrics['audio']['physics']['pitch_variation']*2, 100))
        draw_metric("Pause Ratio", f"{metrics['audio']['physics']['pause_ratio']}%", 100 - metrics['audio']['physics']['pause_ratio']) # Inverse is better

        # Visual Metrics
        if metrics['visual']['is_slide_mode'] and metrics['slides']:
             draw_metric("Slide Readability", f"{int(metrics['slides']['readability_score'])}/100", metrics['slides']['readability_score'])
             draw_metric("Text Density", f"{metrics['slides']['avg_words_per_slide']} words", max(100 - metrics['slides']['avg_words_per_slide'], 0))
        else:
             draw_metric("Eye Contact", f"{metrics['visual']['eye_contact_score']}%", metrics['visual']['eye_contact_score'])
             draw_metric("Posture Stability", f"{metrics['visual']['posture_score']}%", metrics['visual']['posture_score'])

        # --- 5. AI FEEDBACK SECTION ---
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "AI Generated Feedback & Recommendations", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font("Arial", "", 11)
        for item in feedback:
            pdf.cell(10, 8, ">>", 0, 0) 
            pdf.multi_cell(0, 8, f"{item.encode('latin-1', 'ignore').decode('latin-1')}")

        # Footer
        pdf.set_y(-20)
        pdf.set_font("Arial", "I", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, "Generated by ModalX | Daffodil International University", 0, 0, 'C')

        # Output
        temp = f"temp_{s_id}.pdf"
        pdf.output(temp)
        with open(temp, "rb") as f: data = f.read()
        if os.path.exists(temp): os.remove(temp)
        return data

    def analyze(self, path, name, sid, is_url=False):
        vid_path = path
        if is_url: vid_path = VideoDownloader.download_from_url(path)
        if not vid_path: return None

        aud_res = self.audio.process_audio(vid_path)
        vis_res = self.visual.analyze_video(vid_path)
        
        slide_res = None
        if vis_res and vis_res['is_slide_mode']:
            slide_res = self.slides.analyze_slides(vid_path)

        score, feedback = self.calculate_score(aud_res, vis_res, slide_res)
        metrics = {"audio": aud_res, "visual": vis_res, "slides": slide_res}
        
        report = self.generate_pdf(metrics, score, feedback, name, sid)

        if is_url and os.path.exists(vid_path): os.remove(vid_path)
        return {"score": score, "metrics": metrics, "feedback": feedback, "report": report}
