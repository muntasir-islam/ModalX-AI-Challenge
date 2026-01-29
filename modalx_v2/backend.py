"""
ModalX v2 - Unified Deep Learning Backend
Integrates all deep ML models for comprehensive presentation analysis
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import cv2
import whisper
import yt_dlp
import gdown
from moviepy.editor import VideoFileClip
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import deep learning models
from models.transformer_emotion import EmotionAnalyzerV2
from models.action_unit_detector import ActionUnitDetector
from models.gesture_stgcn import GestureAnalyzer
from models.prosody_analyzer import ProsodyAnalyzerV2
from models.content_bert import ContentAnalyzer
from models.slide_vit import SlideAnalyzer

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"ModalX v2 Backend initialized on: {DEVICE}")

# Scoring weights
WEIGHTS = {
    'AUDIO': 0.20,      # 20% - Speech clarity and prosody
    'VISUAL': 0.20,     # 20% - Body language (AU + Gestures)
    'EMOTION': 0.20,    # 20% - Emotional intelligence
    'CONTENT': 0.20,    # 20% - Content quality
    'SLIDES': 0.20      # 20% - Slide design (if applicable)
}


class VideoDownloader:
    """Download videos from YouTube or Google Drive"""
    
    @staticmethod
    def download_from_url(url: str, output_filename: str = "input_video.mp4") -> Optional[str]:
        """Download video from URL"""
        if "drive.google.com" in url:
            try:
                download_path = gdown.download(url, output_filename, quiet=False, fuzzy=True)
                return download_path if download_path else None
            except Exception as e:
                print(f"Google Drive download error: {e}")
                return None
        
        # YouTube or other sources
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_filename,
            'quiet': True,
            'overwrites': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return output_filename if os.path.exists(output_filename) else None
        except Exception as e:
            print(f"Video download error: {e}")
            return None


class SpeechTranscriber:
    """Whisper-based speech transcription"""
    
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size, device=DEVICE)
    
    def transcribe(self, video_path: str) -> Tuple[str, float, float]:
        """
        Transcribe audio from video
        
        Returns:
            transcript: full text
            duration: audio duration in seconds
            wpm: words per minute
        """
        temp_audio = "temp_transcribe_audio.wav"
        
        try:
            # Extract audio
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.audio.write_audiofile(temp_audio, codec='pcm_s16le', verbose=False, logger=None)
            clip.close()
            
            # Transcribe
            result = self.model.transcribe(temp_audio)
            transcript = result["text"]
            
            # Calculate WPM
            words = transcript.split()
            wpm = round(len(words) / (duration / 60), 1) if duration > 0 else 0
            
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return transcript, duration, wpm
            
        except Exception as e:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            print(f"Transcription error: {e}")
            return "", 0, 0


class ModalXSystemV2:
    """
    Unified Deep Learning Presentation Analysis System
    
    Integrates:
    - Transformer Emotion Recognition
    - Facial Action Unit Detection
    - Gesture Recognition (ST-GCN)
    - Deep Prosody Analysis
    - Content Quality (BERT)
    - Slide Design (ViT)
    """
    
    def __init__(self, weights_dir: str = "weights"):
        self.weights_dir = weights_dir
        self.device = DEVICE
        
        print("Initializing ModalX v2 Deep Learning Models...")
        
        # Speech transcription
        self.transcriber = SpeechTranscriber()
        
        # Deep learning models
        self.emotion_analyzer = EmotionAnalyzerV2(
            model_path=self._get_weight_path("emotion_transformer.pt"),
            device=self.device
        )
        
        self.au_detector = ActionUnitDetector(
            model_path=self._get_weight_path("au_detector.pt"),
            device=self.device
        )
        
        self.gesture_analyzer = GestureAnalyzer(
            model_path=self._get_weight_path("gesture_stgcn.pt"),
            device=self.device
        )
        
        self.prosody_analyzer = ProsodyAnalyzerV2(
            model_path=self._get_weight_path("prosody_model.pt"),
            device=self.device
        )
        
        self.content_analyzer = ContentAnalyzer(
            model_path=self._get_weight_path("content_bert.pt"),
            device=self.device
        )
        
        self.slide_analyzer = SlideAnalyzer(
            model_path=self._get_weight_path("slide_vit.pt"),
            device=self.device
        )
        
        print("All models initialized successfully!")
    
    def _get_weight_path(self, filename: str) -> str:
        """Get path to model weights file"""
        return os.path.join(self.weights_dir, filename)
    
    def detect_presentation_mode(self, video_path: str) -> str:
        """Detect if video is face-cam or screen-share"""
        cap = cv2.VideoCapture(video_path)
        face_frames = 0
        total_frames = 0
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        while cap.isOpened() and total_frames < 100:
            ret, frame = cap.read()
            if not ret:
                break
            
            if total_frames % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    face_frames += 1
            
            total_frames += 1
        
        cap.release()
        
        face_ratio = face_frames / max(total_frames // 5, 1)
        return "face_presentation" if face_ratio > 0.5 else "slide_presentation"
    
    def analyze(
        self,
        video_path: str,
        student_name: str,
        student_id: str,
        is_url: bool = False
    ) -> Optional[Dict]:
        """
        Complete presentation analysis
        
        Args:
            video_path: path or URL to video
            student_name: student's name
            student_id: student's ID
            is_url: whether video_path is a URL
            
        Returns:
            Dictionary with all analysis results
        """
        # Download if URL
        if is_url:
            video_path = VideoDownloader.download_from_url(video_path)
            if not video_path:
                return None
        
        try:
            # Detect presentation mode
            mode = self.detect_presentation_mode(video_path)
            print(f"Detected mode: {mode}")
            
            # 1. Speech Transcription
            print("Transcribing speech...")
            transcript, duration, wpm = self.transcriber.transcribe(video_path)
            
            # 2. Emotion Analysis
            print("Analyzing emotions...")
            emotion_timeline, emotion_list, emotion_summary = self.emotion_analyzer.predict(video_path)
            
            # 3. Prosody Analysis
            print("Analyzing prosody...")
            prosody_results = self.prosody_analyzer.analyze(video_path)
            prosody_score, prosody_feedback = self.prosody_analyzer.calculate_score(prosody_results)
            
            # 4. Content Analysis
            print("Analyzing content...")
            content_results = self.content_analyzer.analyze(transcript)
            content_score, content_feedback = self.content_analyzer.calculate_score(content_results)
            
            # 5. Visual Analysis (depends on mode)
            if mode == "face_presentation":
                # Action Unit Detection
                print("Analyzing facial expressions...")
                au_results = self.au_detector.analyze_video(video_path)
                au_score, au_feedback = self.au_detector.calculate_presentation_score(au_results)
                
                # Gesture Analysis
                print("Analyzing gestures...")
                gesture_results = self.gesture_analyzer.analyze_video(video_path)
                gesture_score, gesture_feedback = self.gesture_analyzer.calculate_score(gesture_results)
                
                visual_score = (au_score + gesture_score) / 2
                visual_feedback = au_feedback + gesture_feedback
                slide_results = None
                slide_score = 0
            else:
                # Slide Analysis
                print("Analyzing slides...")
                slide_results = self.slide_analyzer.analyze_video(video_path)
                slide_score, slide_feedback = self.slide_analyzer.calculate_score(slide_results)
                
                visual_score = 50  # Neutral for slide mode
                visual_feedback = []
                au_results = None
                gesture_results = None
            
            # 6. Calculate Final Score
            emotion_score = self._calculate_emotion_score(emotion_summary)
            
            if mode == "face_presentation":
                final_score = (
                    prosody_score * WEIGHTS['AUDIO'] +
                    visual_score * WEIGHTS['VISUAL'] +
                    emotion_score * WEIGHTS['EMOTION'] +
                    content_score * WEIGHTS['CONTENT'] +
                    visual_score * WEIGHTS['SLIDES']  # Use visual for both in face mode
                )
            else:
                final_score = (
                    prosody_score * WEIGHTS['AUDIO'] +
                    50 * WEIGHTS['VISUAL'] +  # Neutral
                    emotion_score * WEIGHTS['EMOTION'] +
                    content_score * WEIGHTS['CONTENT'] +
                    slide_score * WEIGHTS['SLIDES']
                )
            
            # Compile all feedback
            all_feedback = prosody_feedback + content_feedback + visual_feedback
            if mode == "slide_presentation" and slide_results:
                all_feedback.extend(slide_feedback)
            
            # 7. Generate PDF Report
            print("Generating PDF report...")
            report = self.generate_pdf_report(
                student_name=student_name,
                student_id=student_id,
                final_score=int(final_score),
                mode=mode,
                metrics={
                    'audio': {
                        'transcript': transcript[:500] + "..." if len(transcript) > 500 else transcript,
                        'wpm': wpm,
                        'duration': duration,
                        'prosody': prosody_results
                    },
                    'emotion': {
                        'timeline': emotion_timeline,
                        'emotions': emotion_list,
                        'summary': emotion_summary,
                        'score': emotion_score
                    },
                    'content': content_results,
                    'visual': {
                        'au': au_results if mode == "face_presentation" else None,
                        'gesture': gesture_results if mode == "face_presentation" else None
                    },
                    'slides': slide_results
                },
                feedback=all_feedback
            )
            
            # Cleanup
            if is_url and os.path.exists(video_path):
                os.remove(video_path)
            
            return {
                'score': int(final_score),
                'mode': mode,
                'metrics': {
                    'audio': {
                        'transcript': transcript,
                        'wpm': wpm,
                        'duration': duration,
                        'prosody': prosody_results,
                        'prosody_score': prosody_score
                    },
                    'visual': {
                        'au': au_results,
                        'gesture': gesture_results,
                        'score': visual_score
                    },
                    'slides': slide_results,
                    'content': content_results
                },
                'emotion_data': {
                    'times': emotion_timeline,
                    'emotions': emotion_list,
                    'summary': emotion_summary,
                    'score': emotion_score
                },
                'feedback': all_feedback,
                'report': report
            }
            
        except Exception as e:
            print(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            if is_url and os.path.exists(video_path):
                os.remove(video_path)
            return None
    
    def _calculate_emotion_score(self, emotion_summary: Dict) -> float:
        """Calculate emotion score from summary"""
        if not emotion_summary or 'error' in emotion_summary:
            return 50
        
        positive = emotion_summary.get('happy', 0) + \
                   emotion_summary.get('neutral', 0) * 0.8 + \
                   emotion_summary.get('surprise', 0) * 0.5 + \
                   emotion_summary.get('calm', 0) * 0.9
        
        negative = emotion_summary.get('sad', 0) + \
                   emotion_summary.get('fear', 0) + \
                   emotion_summary.get('angry', 0) + \
                   emotion_summary.get('disgust', 0)
        
        total = sum(emotion_summary.values())
        if total == 0:
            return 50
        
        score = (positive / total) * 100
        return round(score, 1)
    
    def generate_pdf_report(
        self,
        student_name: str,
        student_id: str,
        final_score: int,
        mode: str,
        metrics: Dict,
        feedback: List[str]
    ) -> bytes:
        """Generate comprehensive PDF report"""
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_fill_color(24, 40, 72)
        pdf.rect(0, 0, 210, 45, 'F')
        pdf.set_font("Arial", "B", 24)
        pdf.set_text_color(255, 255, 255)
        pdf.set_y(12)
        pdf.cell(0, 10, "ModalX v2.0", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, "Deep Learning Presentation Analysis Report", ln=True, align="C")
        
        # Student Info
        pdf.set_text_color(0, 0, 0)
        pdf.set_y(55)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"Student: {student_name}", ln=True)
        pdf.cell(0, 8, f"ID: {student_id}", ln=True)
        pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.cell(0, 8, f"Mode: {mode.replace('_', ' ').title()}", ln=True)
        
        # Final Score
        pdf.ln(5)
        grade = self._get_grade(final_score)
        pdf.set_font("Arial", "B", 20)
        pdf.set_fill_color(*self._get_grade_color(grade))
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 15, f"Final Score: {final_score}/100 ({grade})", ln=True, align="C", fill=True)
        
        # Score Breakdown
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "1. Score Breakdown", ln=True)
        
        pdf.set_font("Arial", "", 10)
        audio = metrics['audio']
        pdf.cell(50, 8, "Audio (20%)", 1)
        pdf.cell(0, 8, f"WPM: {audio['wpm']}, Prosody: {audio.get('prosody_score', 'N/A')}", 1, 1)
        
        emotion = metrics['emotion']
        pdf.cell(50, 8, "Emotion (20%)", 1)
        pdf.cell(0, 8, f"Score: {emotion.get('score', 'N/A')}", 1, 1)
        
        content = metrics['content']
        pdf.cell(50, 8, "Content (20%)", 1)
        pdf.cell(0, 8, f"Vocab: {content.get('vocabulary_level', 'N/A')}, Args: {content.get('claim_indicators', 0)}", 1, 1)
        
        if mode == "face_presentation":
            visual = metrics['visual']
            pdf.cell(50, 8, "Visual (40%)", 1)
            pdf.cell(0, 8, f"Face + Gesture combined score", 1, 1)
        else:
            slides = metrics['slides']
            if slides:
                pdf.cell(50, 8, "Slides (20%)", 1)
                pdf.cell(0, 8, f"Analyzed: {slides.get('slides_analyzed', 0)} slides", 1, 1)
        
        # Emotion Chart
        if emotion.get('times') and emotion.get('emotions'):
            pdf.ln(10)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "2. Emotional Timeline", ln=True)
            
            try:
                fig, ax = plt.subplots(figsize=(7, 2.5))
                unique_emotions = sorted(list(set(emotion['emotions'])))
                y_vals = [unique_emotions.index(e) for e in emotion['emotions']]
                ax.plot(emotion['times'], y_vals, marker='o', markersize=3, linewidth=1)
                ax.set_yticks(range(len(unique_emotions)))
                ax.set_yticklabels(unique_emotions)
                ax.set_xlabel("Time (seconds)")
                ax.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                chart_path = "temp_emotion_chart.png"
                plt.savefig(chart_path, dpi=100)
                plt.close()
                
                pdf.image(chart_path, x=15, w=180)
                os.remove(chart_path)
            except Exception as e:
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 8, f"(Chart generation error: {e})", ln=True)
        
        # AI Recommendations
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "3. AI Recommendations", ln=True)
        
        pdf.set_font("Arial", "", 10)
        for fb in feedback[:10]:  # Limit to 10 items
            pdf.multi_cell(0, 6, f"• {fb}")
        
        # Output PDF
        temp_pdf = f"temp_report_{student_id}.pdf"
        pdf.output(temp_pdf)
        
        with open(temp_pdf, "rb") as f:
            data = f.read()
        
        os.remove(temp_pdf)
        return data
    
    def _get_grade(self, score: int) -> str:
        """Convert score to letter grade"""
        if score >= 85: return "A+"
        elif score >= 80: return "A"
        elif score >= 75: return "A-"
        elif score >= 70: return "B+"
        elif score >= 65: return "B"
        elif score >= 60: return "B-"
        elif score >= 55: return "C+"
        elif score >= 50: return "C"
        elif score >= 45: return "D"
        return "F"
    
    def _get_grade_color(self, grade: str) -> Tuple[int, int, int]:
        """Get RGB color for grade"""
        colors = {
            "A+": (25, 135, 84), "A": (32, 201, 151), "A-": (13, 202, 240),
            "B+": (255, 193, 7), "B": (253, 126, 20), "B-": (255, 100, 50),
            "C+": (214, 51, 132), "C": (220, 53, 69),
            "D": (180, 50, 50), "F": (150, 30, 30)
        }
        return colors.get(grade, (100, 100, 100))


if __name__ == "__main__":
    # Test initialization
    system = ModalXSystemV2(weights_dir="weights")
    print("ModalX v2 System ready!")
