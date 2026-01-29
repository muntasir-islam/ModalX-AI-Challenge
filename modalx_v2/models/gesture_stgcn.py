"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) for Gesture Recognition
Architecture: MediaPipe Skeleton → Graph Construction → ST-GCN Layers → Gesture Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from collections import deque
import os

# Lazy import for MediaPipe to avoid TensorFlow conflict
mp = None



class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolution Layer"""
    
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        
        self.num_nodes = A.size(0)
        self.register_buffer('A', A)
        
        # Learnable edge importance
        self.edge_importance = nn.Parameter(torch.ones_like(A))
        
        # Convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time, nodes)
        """
        # Apply adjacency with learned importance
        A_hat = self.A * self.edge_importance
        
        # Spatial aggregation
        x = torch.einsum('nctv,vw->nctw', x, A_hat)
        
        # Feature transformation
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        return x


class TemporalConv(nn.Module):
    """Temporal Convolution Block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution Block"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.spatial_conv = SpatialGraphConv(in_channels, out_channels, A, dropout)
        self.temporal_conv = TemporalConv(out_channels, out_channels, stride=stride)
        
        self.residual = residual
        if residual:
            if in_channels != out_channels or stride != 1:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.downsample = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        
        if self.residual:
            x = x + self.downsample(residual)
        
        return F.relu(x)


class GestureSTGCN(nn.Module):
    """
    ST-GCN for Presentation Gesture Recognition
    
    Gesture Classes:
    0: neutral/idle
    1: open_palm (welcoming)
    2: pointing (directing)
    3: counting_fingers (listing)
    4: steepling (confidence)
    5: arms_crossed (defensive)
    6: fidgeting (nervous)
    7: hand_on_face (insecure)
    8: power_pose (authority)
    9: shrug (uncertainty)
    """
    
    GESTURE_CLASSES = [
        'neutral', 'open_palm', 'pointing', 'counting',
        'steepling', 'arms_crossed', 'fidgeting',
        'hand_on_face', 'power_pose', 'shrug'
    ]
    
    GESTURE_SCORES = {
        'neutral': 0,
        'open_palm': 5,
        'pointing': 3,
        'counting': 5,
        'steepling': 4,
        'arms_crossed': -5,
        'fidgeting': -10,
        'hand_on_face': -3,
        'power_pose': 5,
        'shrug': -2
    }
    
    # MediaPipe pose landmark indices for body skeleton
    BODY_EDGES = [
        (0, 1), (0, 4),  # Nose to eyes
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Torso
        (23, 24),  # Hips
        (15, 17), (15, 19), (15, 21),  # Left hand
        (16, 18), (16, 20), (16, 22),  # Right hand
    ]
    
    NUM_KEYPOINTS = 33  # MediaPipe pose landmarks
    
    def __init__(
        self,
        in_channels: int = 3,  # x, y, visibility
        num_classes: int = 10,
        hidden_channels: int = 64,
        num_layers: int = 6,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Build adjacency matrix
        A = self._build_adjacency()
        self.register_buffer('adjacency', A)
        
        # Input processing
        self.input_bn = nn.BatchNorm1d(in_channels * self.NUM_KEYPOINTS)
        
        # ST-GCN layers
        self.layers = nn.ModuleList()
        
        channels = [in_channels, hidden_channels, hidden_channels, hidden_channels * 2,
                   hidden_channels * 2, hidden_channels * 4, hidden_channels * 4]
        strides = [1, 1, 1, 2, 1, 2]
        
        for i in range(num_layers):
            self.layers.append(
                STGCNBlock(
                    channels[i], channels[i + 1],
                    A, stride=strides[i],
                    dropout=dropout
                )
            )
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] // 2, num_classes)
        )
        
        # Movement intensity head
        self.intensity_head = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _build_adjacency(self) -> torch.Tensor:
        """Build skeleton adjacency matrix"""
        A = torch.zeros(self.NUM_KEYPOINTS, self.NUM_KEYPOINTS)
        
        for i, j in self.BODY_EDGES:
            if i < self.NUM_KEYPOINTS and j < self.NUM_KEYPOINTS:
                A[i, j] = 1
                A[j, i] = 1
        
        # Add self-loops
        A = A + torch.eye(self.NUM_KEYPOINTS)
        
        # Normalize
        D = A.sum(dim=1, keepdim=True)
        A = A / D.clamp(min=1)
        
        return A
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, channels, time, num_keypoints) skeleton sequence
            
        Returns:
            logits: (batch, num_classes) gesture classification
            intensity: (batch, 1) movement intensity
        """
        batch_size = x.size(0)
        
        # Input batch norm - use reshape instead of view for non-contiguous tensors
        x = x.contiguous()
        x_flat = x.reshape(batch_size, -1, x.size(2))
        x_flat = self.input_bn(x_flat.permute(0, 2, 1)).permute(0, 2, 1)
        x = x_flat.reshape(batch_size, x.size(1), x.size(2), x.size(3))
        
        # ST-GCN layers
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # (batch, channels)
        
        # Classification
        logits = self.classifier(x)
        intensity = self.intensity_head(x)
        
        return logits, intensity
    
    def predict_gesture(self, skeleton_sequence: torch.Tensor) -> Tuple[str, float, float]:
        """Predict gesture from skeleton sequence"""
        self.eval()
        with torch.no_grad():
            logits, intensity = self.forward(skeleton_sequence)
            probs = F.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            
        return self.GESTURE_CLASSES[pred_idx], probs[0, pred_idx].item(), intensity.item()


class GestureAnalyzer:
    """
    High-level interface for gesture analysis in presentation videos
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Initialize ST-GCN model
        self.model = GestureSTGCN().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded gesture model from {model_path}")
        
        self.model.eval()
        
        # Initialize MediaPipe lazily to avoid TensorFlow conflict
        global mp
        if mp is None:
            import mediapipe as _mp
            mp = _mp
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Sliding window for temporal analysis
        self.window_size = 30  # frames
        self.skeleton_buffer = deque(maxlen=self.window_size)
    
    def extract_skeleton(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract skeleton keypoints from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks as numpy array
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.visibility])
        
        return np.array(landmarks)  # (33, 3)
    
    def analyze_video(self, video_path: str, sample_rate: int = 2) -> Dict:
        """
        Analyze gestures in video
        
        Args:
            video_path: path to video file
            sample_rate: process every Nth frame
            
        Returns:
            Dictionary with gesture statistics and scores
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        gesture_counts = {g: 0 for g in GestureSTGCN.GESTURE_CLASSES}
        gesture_timeline = []
        intensity_values = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                skeleton = self.extract_skeleton(frame)
                
                if skeleton is not None:
                    self.skeleton_buffer.append(skeleton)
                    
                    # Analyze when buffer is full
                    if len(self.skeleton_buffer) == self.window_size:
                        sequence = np.array(list(self.skeleton_buffer))
                        sequence = torch.tensor(sequence, dtype=torch.float32)
                        sequence = sequence.permute(2, 0, 1).unsqueeze(0)  # (1, 3, T, 33)
                        sequence = sequence.to(self.device)
                        
                        gesture, prob, intensity = self.model.predict_gesture(sequence)
                        
                        gesture_counts[gesture] += 1
                        gesture_timeline.append({
                            'time': frame_count / fps,
                            'gesture': gesture,
                            'confidence': prob
                        })
                        intensity_values.append(intensity)
            
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics
        total_gestures = sum(gesture_counts.values())
        gesture_distribution = {
            g: count / max(total_gestures, 1)
            for g, count in gesture_counts.items()
        }
        
        return {
            'gesture_counts': gesture_counts,
            'gesture_distribution': gesture_distribution,
            'timeline': gesture_timeline,
            'avg_movement_intensity': np.mean(intensity_values) if intensity_values else 0,
            'frames_analyzed': frame_count
        }
    
    def calculate_score(self, analysis: Dict) -> Tuple[float, List[str]]:
        """Calculate presentation score based on gestures"""
        feedback = []
        score = 0
        
        distribution = analysis['gesture_distribution']
        
        # Calculate weighted score
        for gesture, weight in GestureSTGCN.GESTURE_SCORES.items():
            score += distribution.get(gesture, 0) * weight * 10
        
        # Base score
        score = 50 + score
        
        # Generate feedback
        if distribution.get('fidgeting', 0) > 0.2:
            feedback.append("High fidgeting detected - try to stay more composed")
        
        if distribution.get('open_palm', 0) > 0.2:
            feedback.append("Great use of open palm gestures - shows openness")
        
        if distribution.get('arms_crossed', 0) > 0.15:
            feedback.append("Arms crossed frequently - may appear defensive")
        
        if distribution.get('power_pose', 0) > 0.1:
            feedback.append("Excellent confident body posture detected")
        
        intensity = analysis['avg_movement_intensity']
        if intensity < 0.2:
            feedback.append("Low movement - try using more dynamic gestures")
        elif intensity > 0.8:
            feedback.append("Very high movement - consider more controlled gestures")
        
        return max(0, min(100, score)), feedback


if __name__ == "__main__":
    # Test the model
    model = GestureSTGCN()
    print(f"ST-GCN parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_skeleton = torch.randn(1, 3, 30, 33)  # (batch, channels, time, keypoints)
    logits, intensity = model(dummy_skeleton)
    print(f"Gesture logits: {logits.shape}")
    print(f"Movement intensity: {intensity.item():.3f}")
