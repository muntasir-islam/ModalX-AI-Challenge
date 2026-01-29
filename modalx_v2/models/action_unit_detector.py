"""
Facial Action Unit (AU) Detection Model
Architecture: ResNet-50 Backbone + Multi-Label AU Classification + Temporal LSTM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os


class ActionUnitClassifier(nn.Module):
    """
    Facial Action Unit Detection using ResNet-50
    
    Detects 17 Action Units commonly used in FACS (Facial Action Coding System)
    """
    
    # Action Unit definitions with interpretations
    AU_DEFINITIONS = {
        'AU1': ('Inner Brow Raiser', 'surprise', 'interest'),
        'AU2': ('Outer Brow Raiser', 'surprise'),
        'AU4': ('Brow Lowerer', 'confusion', 'anger'),
        'AU5': ('Upper Lid Raiser', 'fear', 'surprise'),
        'AU6': ('Cheek Raiser', 'genuine_smile'),
        'AU7': ('Lid Tightener', 'concentration'),
        'AU9': ('Nose Wrinkler', 'disgust'),
        'AU10': ('Upper Lip Raiser', 'disgust'),
        'AU12': ('Lip Corner Puller', 'happiness'),
        'AU14': ('Dimpler', 'contempt'),
        'AU15': ('Lip Corner Depressor', 'sadness'),
        'AU17': ('Chin Raiser', 'uncertainty'),
        'AU20': ('Lip Stretcher', 'fear'),
        'AU23': ('Lip Tightener', 'anger'),
        'AU24': ('Lip Pressor', 'concentration'),
        'AU25': ('Lips Part', 'speaking'),
        'AU26': ('Jaw Drop', 'surprise', 'speaking'),
    }
    
    AU_NAMES = list(AU_DEFINITIONS.keys())
    NUM_AUS = len(AU_NAMES)
    
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        # ResNet-50 backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension from ResNet-50
        self.feature_dim = 2048
        
        # AU Detection Heads (multi-label)
        self.au_heads = nn.ModuleDict({
            au: nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 2)  # Binary classification per AU
            )
            for au in self.AU_NAMES
        })
        
        # Aggregate heads for high-level metrics
        self.engagement_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: (batch, 3, 224, 224) face images
            
        Returns:
            Dictionary with AU predictions and high-level scores
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # (batch, 2048)
        
        # Predict each AU
        au_logits = {}
        for au in self.AU_NAMES:
            au_logits[au] = self.au_heads[au](features)
        
        # High-level metrics
        engagement = self.engagement_head(features)
        confidence = self.confidence_head(features)
        
        return {
            'au_logits': au_logits,
            'engagement': engagement,
            'confidence': confidence,
            'features': features
        }
    
    def predict_aus(self, x: torch.Tensor) -> Dict[str, bool]:
        """Predict active Action Units"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
        active_aus = {}
        for au, logits in outputs['au_logits'].items():
            probs = F.softmax(logits, dim=-1)
            active_aus[au] = probs[0, 1].item() > 0.5
            
        return active_aus


class TemporalAUFusion(nn.Module):
    """
    Temporal fusion of AU predictions across video frames using LSTM
    """
    
    def __init__(self, num_aus: int = 17, hidden_dim: int = 128):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_aus,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Output heads
        self.dominant_emotion = nn.Linear(hidden_dim * 2, 8)
        self.stability_score = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, au_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            au_sequence: (batch, seq_len, num_aus) binary AU activations over time
            
        Returns:
            emotion_logits: (batch, 8) emotion prediction
            stability: (batch, 1) facial stability score
        """
        lstm_out, _ = self.lstm(au_sequence.float())
        
        # Use last output
        final_hidden = lstm_out[:, -1, :]
        
        emotion_logits = self.dominant_emotion(final_hidden)
        stability = torch.sigmoid(self.stability_score(final_hidden))
        
        return emotion_logits, stability


class ActionUnitDetector:
    """
    High-level interface for Action Unit detection in videos
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Initialize models
        self.au_model = ActionUnitClassifier(pretrained=True).to(self.device)
        self.temporal_model = TemporalAUFusion().to(self.device)
        
        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.au_model.load_state_dict(checkpoint['au_model'])
            self.temporal_model.load_state_dict(checkpoint['temporal_model'])
            print(f"Loaded AU detector from {model_path}")
        
        self.au_model.eval()
        self.temporal_model.eval()
        
        # Face detection (using OpenCV cascade for simplicity)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Image transforms
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        """Get image preprocessing transforms"""
        from torchvision import transforms
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect and crop face from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Use largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding
        padding = int(0.2 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face = frame[y1:y2, x1:x2]
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    def analyze_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Analyze a single frame"""
        face = self.detect_face(frame)
        if face is None:
            return None
        
        # Preprocess
        face_tensor = self.transform(face).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.au_model(face_tensor)
        
        # Get active AUs
        active_aus = []
        au_probs = {}
        for au, logits in outputs['au_logits'].items():
            prob = F.softmax(logits, dim=-1)[0, 1].item()
            au_probs[au] = prob
            if prob > 0.5:
                active_aus.append(au)
        
        return {
            'active_aus': active_aus,
            'au_probabilities': au_probs,
            'engagement': outputs['engagement'].item(),
            'confidence': outputs['confidence'].item()
        }
    
    def analyze_video(self, video_path: str, sample_rate: int = 5) -> Dict:
        """
        Analyze entire video for Action Units
        
        Args:
            video_path: path to video file
            sample_rate: analyze every Nth frame
            
        Returns:
            Dictionary with AU statistics and derived metrics
        """
        cap = cv2.VideoCapture(video_path)
        
        frame_results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                result = self.analyze_frame(frame)
                if result:
                    frame_results.append(result)
            
            frame_count += 1
        
        cap.release()
        
        if not frame_results:
            return {
                'au_frequencies': {},
                'avg_engagement': 0,
                'avg_confidence': 0,
                'genuine_smile_ratio': 0,
                'focus_score': 0
            }
        
        # Aggregate results
        au_counts = {au: 0 for au in ActionUnitClassifier.AU_NAMES}
        total_engagement = 0
        total_confidence = 0
        
        for result in frame_results:
            for au in result['active_aus']:
                au_counts[au] += 1
            total_engagement += result['engagement']
            total_confidence += result['confidence']
        
        n_frames = len(frame_results)
        au_frequencies = {au: count / n_frames for au, count in au_counts.items()}
        
        # Derived metrics
        genuine_smile = au_frequencies.get('AU6', 0) * au_frequencies.get('AU12', 0)
        focus = 1.0 - au_frequencies.get('AU26', 0) * 0.5  # Less jaw drop = more focus
        
        return {
            'au_frequencies': au_frequencies,
            'avg_engagement': total_engagement / n_frames,
            'avg_confidence': total_confidence / n_frames,
            'genuine_smile_ratio': genuine_smile,
            'focus_score': focus,
            'frames_analyzed': n_frames
        }
    
    def calculate_presentation_score(self, au_analysis: Dict) -> Tuple[float, List[str]]:
        """
        Calculate presentation score based on AU analysis
        
        Returns:
            score: 0-100 score
            feedback: list of feedback strings
        """
        au_freq = au_analysis['au_frequencies']
        feedback = []
        score = 100
        
        # Positive indicators
        if au_freq.get('AU12', 0) > 0.3:  # Smiling
            score += 5
            feedback.append("Great: Genuine positivity detected in facial expressions")
        
        if au_analysis['genuine_smile_ratio'] > 0.2:
            score += 5
            feedback.append("Excellent: Authentic engagement shown through genuine smiles")
        
        # Negative indicators
        if au_freq.get('AU4', 0) > 0.4:  # Brow lowerer
            score -= 10
            feedback.append("Warning: Frequent frowning detected - may appear stressed")
        
        if au_freq.get('AU15', 0) > 0.3:  # Lip corner depressor
            score -= 10
            feedback.append("Notice: Facial expressions suggest low energy or sadness")
        
        if au_freq.get('AU17', 0) > 0.3:  # Chin raiser
            score -= 5
            feedback.append("Tip: Chin raising suggests uncertainty - project more confidence")
        
        # Engagement metrics
        if au_analysis['avg_engagement'] < 0.5:
            score -= 10
            feedback.append("Improvement needed: Facial engagement appears low")
        
        return max(0, min(100, score)), feedback


if __name__ == "__main__":
    # Test the model
    model = ActionUnitClassifier()
    print(f"AU Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_face = torch.randn(1, 3, 224, 224)
    outputs = model(dummy_face)
    print(f"AU outputs: {len(outputs['au_logits'])} AUs")
    print(f"Engagement: {outputs['engagement'].item():.3f}")
