"""
Transformer-based Speech Emotion Recognition Model
Architecture: Wav2Vec2 Feature Extractor + Multi-Head Self-Attention + Classification Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import Counter
import os


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism for temporal modeling"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(attn_output)


class AttentionPooling(nn.Module):
    """Attention-weighted pooling for sequence aggregation"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, embed_dim)
        weights = F.softmax(self.attention(x), dim=1)  # (batch, seq_len, 1)
        pooled = torch.sum(x * weights, dim=1)  # (batch, embed_dim)
        return pooled


class TransformerEmotionModel(nn.Module):
    """
    Speech Emotion Recognition using Transformer Architecture
    
    Features:
    - CNN-based audio feature extraction (Mel spectrogram)
    - Multi-Head Self-Attention for temporal modeling
    - Attention pooling for sequence aggregation
    - 8-class emotion classification
    """
    
    EMOTIONS = [
        'angry', 'disgust', 'fear', 'happy',
        'neutral', 'sad', 'surprise', 'calm'
    ]
    
    def __init__(
        self,
        n_mels: int = 80,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        num_emotions: int = 8,
        dropout: float = 0.1,
        sample_rate: int = 16000,
        chunk_duration: float = 3.0
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        
        # Mel Spectrogram Transform
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels
        )
        
        # CNN Feature Extractor
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Positional Encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, embed_dim) * 0.02)
        
        # Transformer Layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Attention Pooling
        self.attention_pool = AttentionPooling(embed_dim)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_emotions)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram features from waveform"""
        # waveform: (batch, samples)
        mel = self.mel_spec(waveform)  # (batch, n_mels, time)
        mel = torch.log(mel + 1e-9)  # Log mel spectrogram
        return mel
    
    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            waveform: (batch, samples) audio waveform
            
        Returns:
            logits: (batch, num_emotions) classification logits
            confidence: (batch, 1) confidence score
        """
        # Extract mel features
        mel = self.extract_features(waveform)  # (batch, n_mels, time)
        
        # CNN feature extraction
        features = self.cnn_extractor(mel)  # (batch, embed_dim, time')
        features = features.transpose(1, 2)  # (batch, time', embed_dim)
        
        # Add positional encoding
        seq_len = features.size(1)
        features = features + self.pos_encoding[:, :seq_len, :]
        
        # Transformer layers
        for layer in self.transformer_layers:
            features = layer(features)
        
        # Attention pooling
        pooled = self.attention_pool(features)  # (batch, embed_dim)
        
        # Classification
        logits = self.classifier(pooled)
        confidence = self.confidence_head(pooled)
        
        return logits, confidence
    
    def predict_emotion(self, waveform: torch.Tensor) -> Tuple[str, float, float]:
        """
        Predict emotion from waveform
        
        Returns:
            emotion: predicted emotion string
            probability: softmax probability
            confidence: model confidence
        """
        self.eval()
        with torch.no_grad():
            logits, confidence = self.forward(waveform)
            probs = F.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            pred_prob = probs[0, pred_idx].item()
            
        return self.EMOTIONS[pred_idx], pred_prob, confidence.item()


class EmotionAnalyzerV2:
    """
    High-level interface for emotion analysis on video/audio files
    Uses TransformerEmotionModel for prediction
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.sample_rate = 16000
        self.chunk_duration = 3.0
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Initialize model
        self.model = TransformerEmotionModel().to(self.device)
        
        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded emotion model from {model_path}")
        else:
            print("Using randomly initialized emotion model (no pretrained weights)")
        
        self.model.eval()
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)  # (samples,)
    
    def analyze(self, audio_path: str) -> Tuple[List[float], List[str], Dict[str, int], float]:
        """
        Analyze emotions in audio file
        
        Returns:
            timeline: list of timestamps (seconds)
            emotions: list of detected emotions
            summary: emotion counts
            avg_confidence: average confidence score
        """
        waveform = self.load_audio(audio_path)
        total_samples = waveform.size(0)
        
        timeline = []
        emotions = []
        confidences = []
        
        # Process in chunks
        for start in range(0, total_samples - self.chunk_samples, self.chunk_samples):
            chunk = waveform[start:start + self.chunk_samples]
            
            # Pad if necessary
            if len(chunk) < self.chunk_samples:
                chunk = F.pad(chunk, (0, self.chunk_samples - len(chunk)))
            
            chunk = chunk.unsqueeze(0).to(self.device)
            
            emotion, prob, conf = self.model.predict_emotion(chunk)
            
            timeline.append(start / self.sample_rate)
            emotions.append(emotion)
            confidences.append(conf)
        
        summary = dict(Counter(emotions))
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return timeline, emotions, summary, avg_confidence
    
    def predict(self, video_path: str) -> Tuple[List[float], List[str], Dict[str, int]]:
        """
        Backward-compatible predict method (same interface as original)
        
        Returns:
            timeline_seconds: list of timestamps
            detected_emotions: list of emotions
            summary: emotion distribution
        """
        # Extract audio from video first
        from moviepy.editor import VideoFileClip
        
        temp_audio = "temp_emotion_audio.wav"
        try:
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(temp_audio, fps=self.sample_rate, verbose=False, logger=None)
            clip.close()
            
            timeline, emotions, summary, _ = self.analyze(temp_audio)
            
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return timeline, emotions, summary
            
        except Exception as e:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            return [], [], {"error": str(e)}


if __name__ == "__main__":
    # Test the model
    model = TransformerEmotionModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_audio = torch.randn(1, 48000)  # 3 seconds at 16kHz
    logits, confidence = model(dummy_audio)
    print(f"Output shape: {logits.shape}, Confidence: {confidence.item():.3f}")
