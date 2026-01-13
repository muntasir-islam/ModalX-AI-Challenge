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
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except:
            self.sentiment_pipeline = None

    def analyze_prosody(self, audio_path):
        y, sr = librosa.load(audio_path)
        
        rms = librosa.feature.rms(y=y)[0]
        avg_volume = np.mean(rms)
        vol_score = min(max(avg_volume / 0.05, 0), 1) * 100 
        
        non_silent_intervals = librosa.effects.split(y, top_db=25)
        total_duration = librosa.get_duration(y=y, sr=sr)
        non_silent_duration = sum([(end - start) / sr for start, end in non_silent_intervals])
        
        pause_ratio = 1.0 - (non_silent_duration / total_duration) if total_duration > 0 else 0
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        mask = magnitudes > np.median(magnitudes)
        active_pitches = pitches[mask]
        valid_pitches = active_pitches[(active_pitches > 70) & (active_pitches < 400)]
        
        pitch_std = np.std(valid_pitches) if len(valid_pitches) > 0 else 0
        
        return {
            "volume_score": round(vol_score, 1),
            "pause_ratio": round(pause_ratio * 100, 1),
            "pitch_variation": round(float(pitch_std), 2)
        }

    def process_audio(self, video_path):
        audio_path = "temp_audio.wav"
        try:
            video = VideoFileClip(video_path)
            duration = video.duration
            video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
            video.close()
        except Exception as e:
            return None

        try:
            result = self.whisper_model.transcribe(audio_path)
            transcript = result["text"]
            
            words = transcript.split()
            wpm = round(len(words) / (duration / 60), 1) if duration > 0 else 0
            
            fillers = ["um", "uh", "like", "actually", "basically", "literally", "so"]
            filler_count = sum(1 for w in words if re.sub(r'[^\w]', '', w.lower()) in fillers)
            
            physics = self.analyze_prosody(audio_path)
            
            stopwords = set(["the", "and", "is", "of", "to", "a", "in", "that", "it", "for", "on", "with", "as", "was", "at", "um", "uh", "like", "actually", "so", "you", "my"])
            clean_words = [w.lower() for w in re.findall(r'\b\w+\b', transcript) if len(w) > 3]
            word_counts = {}
            for w in clean_words:
                if w not in stopwords:
                    word_counts[w] = word_counts.get(w, 0) + 1
            topics = [k.title() for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:5]]

            if os.path.exists(audio_path): os.remove(audio_path)

            return {
                "transcript": transcript,
                "wpm": wpm,
                "filler_count": filler_count,
                "physics": physics,
                "topics": topics
            }

        except Exception as e:
            if os.path.exists(audio_path): os.remove(audio_path)
            return None

class VisualAnalyzer:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = 0
        analyzed_frames = 0
        eye_contact_frames = 0
        good_posture_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if total_frames % 10 == 0:
                analyzed_frames += 1
                try:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.holistic.process(image)
                    
                    if results.face_landmarks:
                        nose_x = results.face_landmarks.landmark[1].x
                        left_cheek_x = results.face_landmarks.landmark[234].x
                        right_cheek_x = results.face_landmarks.landmark[454].x
                        face_center = (left_cheek_x + right_cheek_x) / 2
                        if abs(nose_x - face_center) < 0.05:
                            eye_contact_frames += 1

                    if results.pose_landmarks:
                        l_shldr_y = results.pose_landmarks.landmark[11].y
                        r_shldr_y = results.pose_landmarks.landmark[12].y
                        if abs(l_shldr_y - r_shldr_y) < 0.05:
                            good_posture_frames += 1
                except:
                    pass
            total_frames += 1

        cap.release()
        
        if analyzed_frames == 0: return {"eye_contact_score": 0, "posture_score": 0}

        return {
            "eye_contact_score": round((eye_contact_frames / analyzed_frames) * 100, 1),
            "posture_score": round((good_posture_frames / analyzed_frames) * 100, 1),
            "hand_score": 50
        }

class ModalXSystem:
    def __init__(self):
        self.audio_engine = SpeechAnalyzer()
        self.visual_engine = VisualAnalyzer()

    def calculate_score(self, audio, visual):
        score = 100
        feedback = []

        pause_ratio = audio['physics']['pause_ratio']
        wpm = audio['wpm']
        
        if wpm < 110:
            if pause_ratio > 30:
                score -= 10
                feedback.append("Frequent hesitations detected. Try to flow more smoothly.")
            else:
                score -= 5
                feedback.append("Speaking pace is slow. Try to increase energy.")
        elif wpm > 170:
            score -= 10
            feedback.append("Speaking very fast. Ensure you aren't rushing.")
        else:
            feedback.append("Good speaking pace.")

        vol = audio['physics']['volume_score']
        if vol < 30:
            score -= 10
            feedback.append("Volume is low. Project your voice for better confidence.")
        
        pitch_var = audio['physics']['pitch_variation']
        if pitch_var < 15:
            score -= 10
            feedback.append("Voice sounds monotone. Vary your pitch to keep interest.")
        
        if audio['filler_count'] > 6:
            score -= 5
            feedback.append(f"Detected {audio['filler_count']} filler words. Pause instead.")

        if visual['eye_contact_score'] < 60:
            score -= 10
            feedback.append("Low eye contact. Look at the camera.")
        
        if visual['posture_score'] < 80:
            score -= 5
            feedback.append("Improve posture. Keep shoulders steady.")

        return max(0, score), feedback

    def generate_pdf(self, metrics, score, feedback, student_name, student_id):
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 20, "Presentation Assessment Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, "DIU Smart Faculty Grader", ln=True, align="C")
        pdf.line(10, 35, 200, 35)
        
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, f"Student Name:", border=0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"{student_name}", ln=True)
        
        pdf.cell(40, 10, f"Student ID:", border=0)
        pdf.cell(0, 10, f"{student_id}", ln=True)
        
        pdf.cell(40, 10, f"Date:", border=0)
        pdf.cell(0, 10, f"{datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

        letter_grade = "F"
        if score >= 80: letter_grade = "A+ (Outstanding)"
        elif score >= 75: letter_grade = "A (Excellent)"
        elif score >= 70: letter_grade = "A- (Very Good)"
        elif score >= 65: letter_grade = "B+ (Good)"
        elif score >= 60: letter_grade = "B (Satisfactory)"
        else: letter_grade = "C (Needs Improvement)"

        pdf.ln(10)
        pdf.set_fill_color(240, 240, 240)
        pdf.rect(10, pdf.get_y(), 190, 30, 'F')
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 15, f"Final Score: {score}/100", ln=True, align="C")
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 15, f"Grade: {letter_grade}", ln=True, align="C")
        
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Performance Metrics:", ln=True)
        pdf.set_font("Arial", "", 12)
        
        pdf.cell(100, 10, "Metric", border=1)
        pdf.cell(90, 10, "Result", border=1, ln=True)
        
        pdf.cell(100, 10, "Speaking Pace (WPM)", border=1)
        pdf.cell(90, 10, f"{metrics['audio']['wpm']}", border=1, ln=True)
        
        pdf.cell(100, 10, "Pitch Variation", border=1)
        pdf.cell(90, 10, f"{metrics['audio']['physics']['pitch_variation']}", border=1, ln=True)
        
        pdf.cell(100, 10, "Eye Contact", border=1)
        pdf.cell(90, 10, f"{metrics['visual']['eye_contact_score']}%", border=1, ln=True)
        
        pdf.ln(10)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Faculty Feedback:", ln=True)
        pdf.set_font("Arial", "", 12)
        for tip in feedback:
            clean_tip = tip.encode('latin-1', 'ignore').decode('latin-1')
            pdf.multi_cell(0, 8, f"- {clean_tip}")

        temp_filename = f"temp_{student_id}_{datetime.now().timestamp()}.pdf"
        pdf.output(temp_filename)
        
        with open(temp_filename, "rb") as f:
            pdf_bytes = f.read()
            
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return pdf_bytes

    def analyze(self, input_path, student_name="Unknown", student_id="000", is_url=False):
        video_path = input_path
        if is_url:
            video_path = VideoDownloader.download_from_url(input_path)
            if not video_path: return None

        audio_res = self.audio_engine.process_audio(video_path)
        visual_res = self.visual_engine.analyze_video(video_path)

        if not audio_res: return None
        if not visual_res: visual_res = {"eye_contact_score": 0, "posture_score": 0}

        score, feedback = self.calculate_score(audio_res, visual_res)
        all_metrics = {"audio": audio_res, "visual": visual_res}
        
        report_bytes = self.generate_pdf(all_metrics, score, feedback, student_name, student_id)

        if is_url and os.path.exists(video_path): os.remove(video_path)

        return {
            "score": score,
            "metrics": all_metrics,
            "feedback": feedback,
            "report": report_bytes
        }
