"""
CNN-BiLSTM Prosody Analyzer for Deep Speech Analysis
Architecture: 1D CNN Feature Extractor → BiLSTM Sequence Modeling → Multi-Task Heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from typing import Dict, Tuple, Optional, List
import os


class Conv1DBlock(nn.Module):
    """1D Convolutional Block with BatchNorm and Residual Connection"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1, stride) if in_channels != out_channels or stride != 1 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)


class ProsodyEncoder(nn.Module):
    """CNN encoder for prosodic feature extraction"""
    
    def __init__(self, input_dim: int = 80, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            Conv1DBlock(input_dim, 128, kernel_size=5, dropout=dropout),
            nn.MaxPool1d(2),
            
            Conv1DBlock(128, 256, kernel_size=5, dropout=dropout),
            nn.MaxPool1d(2),
            
            Conv1DBlock(256, hidden_dim, kernel_size=3, dropout=dropout),
            nn.MaxPool1d(2),
            
            Conv1DBlock(hidden_dim, hidden_dim, kernel_size=3, dropout=dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim, time)
        return self.encoder(x)  # (batch, hidden_dim, time')


class ProsodyAnalyzer(nn.Module):
    """
    CNN-BiLSTM model for deep prosody analysis
    
    Multi-task outputs:
    - Pitch dynamism score (0-100)
    - Energy consistency score (0-100)
    - Speaking rate variation (0-100)
    - Fluency index (0-100)
    - Engagement prediction (0-100)
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        sample_rate: int = 16000
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        
        # Mel spectrogram
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels
        )
        
        # CNN Encoder
        self.encoder = ProsodyEncoder(n_mels, hidden_dim, dropout)
        
        # BiLSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Multi-task heads
        self.pitch_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.fluency_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.engagement_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Frame-level pitch contour prediction
        self.pitch_contour_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Relative pitch
        )
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log-mel spectrogram"""
        mel = self.mel_spec(waveform)
        log_mel = torch.log(mel + 1e-9)
        return log_mel
    
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            waveform: (batch, samples) audio waveform
            
        Returns:
            Dictionary with prosody metrics
        """
        # Extract features
        mel = self.extract_features(waveform)  # (batch, n_mels, time)
        
        # CNN encoding
        encoded = self.encoder(mel)  # (batch, hidden, time')
        
        # LSTM sequence modeling
        lstm_in = encoded.transpose(1, 2)  # (batch, time', hidden)
        lstm_out, _ = self.lstm(lstm_in)  # (batch, time', hidden)
        
        # Global average pooling
        pooled = lstm_out.mean(dim=1)  # (batch, hidden)
        
        # Multi-task outputs
        results = {
            'pitch_dynamism': self.pitch_head(pooled) * 100,
            'energy_consistency': self.energy_head(pooled) * 100,
            'rate_variation': self.rate_head(pooled) * 100,
            'fluency_index': self.fluency_head(pooled) * 100,
            'engagement': self.engagement_head(pooled) * 100,
            'pitch_contour': self.pitch_contour_head(lstm_out),  # Frame-level
        }
        
        return results


class ProsodyAnalyzerV2:
    """
    High-level interface for prosody analysis
    Combines deep learning model with traditional signal processing
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.sample_rate = 16000
        
        # Initialize model
        self.model = ProsodyAnalyzer(sample_rate=self.sample_rate).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded prosody model from {model_path}")
        
        self.model.eval()
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)
    
    def analyze_traditional(self, audio_path: str) -> Dict[str, float]:
        """Traditional prosody analysis using librosa"""
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_values = pitch_values[pitch_values > 0]
        
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        pitch_range = np.ptp(pitch_values) if len(pitch_values) > 0 else 0
        
        # Energy analysis
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        
        # Pause analysis
        intervals = librosa.effects.split(y, top_db=25)
        speaking_time = sum([end - start for start, end in intervals]) / sr
        total_time = len(y) / sr
        pause_ratio = 1.0 - (speaking_time / total_time) if total_time > 0 else 0
        
        # Speaking rate variability (local tempo)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_local = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        tempo_var = np.std(tempo_local) if len(tempo_local) > 0 else 0
        
        return {
            'pitch_mean': round(pitch_mean, 2),
            'pitch_std': round(pitch_std, 2),
            'pitch_range': round(pitch_range, 2),
            'energy_mean': round(float(energy_mean), 4),
            'energy_std': round(float(energy_std), 4),
            'pause_ratio': round(pause_ratio * 100, 1),
            'tempo_variability': round(float(tempo_var), 2)
        }
    
    def analyze(self, video_path: str) -> Dict:
        """
        Complete prosody analysis
        
        Args:
            video_path: path to video/audio file
            
        Returns:
            Dictionary with all prosody metrics
        """
        from moviepy.editor import VideoFileClip
        
        temp_audio = "temp_prosody_audio.wav"
        
        try:
            # Extract audio
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(temp_audio, fps=self.sample_rate, verbose=False, logger=None)
            clip.close()
            
            # Deep learning analysis
            waveform = self.load_audio(temp_audio)
            waveform = waveform.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                dl_results = self.model(waveform)
            
            # Traditional analysis
            trad_results = self.analyze_traditional(temp_audio)
            
            # Combine results
            combined = {
                # Deep learning scores
                'pitch_dynamism': dl_results['pitch_dynamism'].item(),
                'energy_consistency': dl_results['energy_consistency'].item(),
                'rate_variation': dl_results['rate_variation'].item(),
                'fluency_index': dl_results['fluency_index'].item(),
                'engagement': dl_results['engagement'].item(),
                
                # Traditional metrics
                'pitch_hz': trad_results['pitch_mean'],
                'pitch_std_hz': trad_results['pitch_std'],
                'pause_ratio_pct': trad_results['pause_ratio'],
                'energy': trad_results['energy_mean'],
                'tempo_var': trad_results['tempo_variability']
            }
            
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            return combined
            
        except Exception as e:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            return {'error': str(e)}
    
    def calculate_score(self, analysis: Dict) -> Tuple[float, List[str]]:
        """Calculate presentation score based on prosody"""
        feedback = []
        
        if 'error' in analysis:
            return 50, ["Could not analyze audio prosody"]
        
        # Calculate composite score
        scores = []
        
        # Pitch dynamism (monotone is bad)
        pitch_score = analysis.get('pitch_dynamism', 50)
        scores.append(pitch_score)
        if pitch_score < 40:
            feedback.append("Voice sounds monotone - vary your pitch more")
        elif pitch_score > 80:
            feedback.append("Excellent vocal variety and expressiveness")
        
        # Fluency (fewer pauses/hesitations is better)
        fluency = analysis.get('fluency_index', 50)
        scores.append(fluency)
        if fluency < 40:
            feedback.append("Speech has many pauses - practice for smoother delivery")
        elif fluency > 80:
            feedback.append("Very smooth and fluent speech delivery")
        
        # Energy consistency
        energy = analysis.get('energy_consistency', 50)
        scores.append(energy)
        if energy < 40:
            feedback.append("Volume varies too much - maintain consistent energy")
        
        # Rate variation (some is good, too much is bad)
        rate = analysis.get('rate_variation', 50)
        if rate < 30:
            feedback.append("Speaking pace is very uniform - add some variation")
        elif rate > 70:
            feedback.append("Speaking pace is erratic - try to be more consistent")
        
        # Overall engagement
        engagement = analysis.get('engagement', 50)
        scores.append(engagement)
        
        final_score = np.mean(scores)
        
        return round(final_score, 1), feedback


if __name__ == "__main__":
    # Test the model
    model = ProsodyAnalyzer()
    print(f"Prosody model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_audio = torch.randn(1, 48000)  # 3 seconds
    results = model(dummy_audio)
    
    print("Prosody outputs:")
    for k, v in results.items():
        if v.dim() <= 2:
            print(f"  {k}: {v.mean().item():.2f}")
