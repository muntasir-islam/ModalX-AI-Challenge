"""
Vision Transformer (ViT) for Slide Quality Analysis
Architecture: ViT Encoder + Multi-Label Classification Heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import pytesseract
from PIL import Image


class SlideViT(nn.Module):
    """
    Vision Transformer for Slide Design Quality Analysis
    
    Multi-task outputs:
    - Text density score (0-100)
    - Visual balance score (0-100)
    - Color harmony score (0-100)
    - Chart appropriateness (none/appropriate/inappropriate)
    - Overall design grade (A-F)
    """
    
    DESIGN_GRADES = ['A', 'B', 'C', 'D', 'F']
    CHART_CLASSES = ['no_chart', 'appropriate', 'inappropriate']
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        hidden_dim: int = 768,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Load pretrained ViT
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Feature dimension from ViT
        self.feature_dim = self.vit.num_features
        
        # Shared projection
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.text_density_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.visual_balance_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.color_harmony_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.chart_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 classes
        )
        
        self.grade_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5)  # 5 grades
        )
        
        # Visual complexity score
        self.complexity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: (batch, 3, 224, 224) slide images
            
        Returns:
            Dictionary with all scores
        """
        # ViT encoding
        features = self.vit(x)  # (batch, feature_dim)
        
        # Project
        projected = self.projector(features)  # (batch, hidden_dim)
        
        # Multi-task outputs
        return {
            'text_density': self.text_density_head(projected) * 100,
            'visual_balance': self.visual_balance_head(projected) * 100,
            'color_harmony': self.color_harmony_head(projected) * 100,
            'chart_logits': self.chart_head(projected),
            'grade_logits': self.grade_head(projected),
            'visual_complexity': self.complexity_head(projected) * 100
        }


class SlideAnalyzer:
    """
    High-level interface for slide analysis
    Combines ViT model with traditional image analysis
    """
    
    # Optimal slide design parameters
    OPTIMAL_WORD_COUNT = (20, 40)  # words per slide
    OPTIMAL_FONT_SIZE_RATIO = 0.03  # Title should be ~3% of slide height
    OPTIMAL_IMAGE_RATIO = (0.3, 0.6)  # 30-60% of slide should be images
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Initialize model
        self.model = SlideViT().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            print(f"Loaded slide model from {model_path}")
        
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from slide image using OCR"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            text = pytesseract.image_to_string(gray)
            return text.strip()
        except:
            return ""
    
    def analyze_color_distribution(self, image: np.ndarray) -> Dict:
        """Analyze color distribution and harmony"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate color histogram
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        # Dominant hues
        dominant_hues = np.argsort(h_hist.flatten())[-3:]
        
        # Saturation statistics
        avg_saturation = np.mean(hsv[:, :, 1])
        
        # Color diversity (entropy)
        h_prob = h_hist / h_hist.sum()
        color_entropy = -np.sum(h_prob * np.log2(h_prob + 1e-10))
        
        # Contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        
        return {
            'dominant_hues': dominant_hues.tolist(),
            'avg_saturation': float(avg_saturation),
            'color_diversity': float(color_entropy),
            'contrast': float(contrast)
        }
    
    def analyze_layout(self, image: np.ndarray) -> Dict:
        """Analyze visual layout and balance"""
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate visual weight in quadrants
        mid_h, mid_w = height // 2, width // 2
        quadrants = [
            gray[:mid_h, :mid_w],      # Top-left
            gray[:mid_h, mid_w:],       # Top-right
            gray[mid_h:, :mid_w],       # Bottom-left
            gray[mid_h:, mid_w:]        # Bottom-right
        ]
        
        weights = [np.mean(255 - q) for q in quadrants]  # Darker = heavier
        
        # Balance score (lower variance = better balance)
        weight_variance = np.var(weights)
        balance_score = max(0, 100 - weight_variance / 10)
        
        # Edge detection for visual complexity
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Text region estimation
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        text_ratio = np.sum(binary > 0) / (height * width)
        
        return {
            'quadrant_weights': weights,
            'balance_score': round(balance_score, 1),
            'edge_density': round(edge_density * 100, 2),
            'text_coverage': round(text_ratio * 100, 2)
        }
    
    def analyze_slide(self, image: np.ndarray) -> Dict:
        """Analyze a single slide image"""
        # Traditional analysis
        text = self.extract_text(image)
        word_count = len(text.split()) if text else 0
        
        color_analysis = self.analyze_color_distribution(image)
        layout_analysis = self.analyze_layout(image)
        
        # Deep learning analysis
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dl_results = self.model(input_tensor)
        
        # Process grade
        grade_probs = F.softmax(dl_results['grade_logits'], dim=-1)
        grade_idx = torch.argmax(grade_probs, dim=-1).item()
        
        # Process chart classification
        chart_probs = F.softmax(dl_results['chart_logits'], dim=-1)
        chart_idx = torch.argmax(chart_probs, dim=-1).item()
        
        return {
            # Deep learning scores
            'text_density_score': dl_results['text_density'].item(),
            'visual_balance_score': dl_results['visual_balance'].item(),
            'color_harmony_score': dl_results['color_harmony'].item(),
            'design_grade': SlideViT.DESIGN_GRADES[grade_idx],
            'grade_confidence': grade_probs[0, grade_idx].item(),
            'chart_status': SlideViT.CHART_CLASSES[chart_idx],
            'visual_complexity': dl_results['visual_complexity'].item(),
            
            # Traditional analysis
            'word_count': word_count,
            'has_too_much_text': word_count > 50,
            **color_analysis,
            **layout_analysis
        }
    
    def analyze_video(self, video_path: str, sample_interval: int = 60) -> Dict:
        """Analyze slides from video"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        slide_results = []
        frame_count = 0
        last_slide_hash = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every N frames
            if frame_count % sample_interval == 0:
                # Simple hash to detect slide changes
                small = cv2.resize(frame, (16, 16))
                current_hash = hash(small.tobytes())
                
                # Only analyze if slide changed significantly
                if current_hash != last_slide_hash:
                    result = self.analyze_slide(frame)
                    result['timestamp'] = frame_count / fps
                    slide_results.append(result)
                    last_slide_hash = current_hash
            
            frame_count += 1
        
        cap.release()
        
        if not slide_results:
            return {
                'slides_analyzed': 0,
                'avg_scores': {},
                'feedback': ['No slides detected in video']
            }
        
        # Aggregate results
        avg_scores = {
            'avg_text_density': np.mean([s['text_density_score'] for s in slide_results]),
            'avg_visual_balance': np.mean([s['visual_balance_score'] for s in slide_results]),
            'avg_color_harmony': np.mean([s['color_harmony_score'] for s in slide_results]),
            'avg_word_count': np.mean([s['word_count'] for s in slide_results]),
            'text_heavy_slides': sum(1 for s in slide_results if s['has_too_much_text'])
        }
        
        # Grade distribution
        grade_counts = {}
        for s in slide_results:
            g = s['design_grade']
            grade_counts[g] = grade_counts.get(g, 0) + 1
        
        return {
            'slides_analyzed': len(slide_results),
            'slide_details': slide_results,
            'avg_scores': avg_scores,
            'grade_distribution': grade_counts
        }
    
    def calculate_score(self, analysis: Dict) -> Tuple[float, List[str]]:
        """Calculate overall slide design score"""
        feedback = []
        
        if analysis.get('slides_analyzed', 0) == 0:
            return 50, feedback
        
        avg = analysis.get('avg_scores', {})
        
        scores = []
        
        # Text density (lower is often better for slides)
        text_score = 100 - min(avg.get('avg_text_density', 50), 100)
        scores.append(text_score)
        
        if avg.get('avg_word_count', 0) > 50:
            feedback.append("Slides have too much text - use bullet points")
        
        # Visual balance
        balance = avg.get('avg_visual_balance', 50)
        scores.append(balance)
        
        if balance < 50:
            feedback.append("Improve visual balance - distribute content evenly")
        
        # Color harmony
        color = avg.get('avg_color_harmony', 50)
        scores.append(color)
        
        if color > 80:
            feedback.append("Excellent color choices in slides")
        
        # Text-heavy penalty
        text_heavy = analysis.get('avg_scores', {}).get('text_heavy_slides', 0)
        if text_heavy > 0:
            scores.append(max(0, 100 - text_heavy * 10))
            feedback.append(f"{text_heavy} slides have excessive text")
        
        # Grade distribution bonus
        grades = analysis.get('grade_distribution', {})
        a_grades = grades.get('A', 0) + grades.get('B', 0)
        total = sum(grades.values()) or 1
        grade_bonus = (a_grades / total) * 20
        
        final_score = np.mean(scores) + grade_bonus
        
        return round(min(100, final_score), 1), feedback


if __name__ == "__main__":
    # Test the model
    model = SlideViT()
    print(f"Slide ViT parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_slide = torch.randn(1, 3, 224, 224)
    outputs = model(dummy_slide)
    
    print("Slide outputs:")
    for k, v in outputs.items():
        if 'logits' in k:
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v.item():.2f}")
