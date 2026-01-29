"""
Content Quality Scorer using DistilBERT
Architecture: DistilBERT Encoder + Multi-Task Classification Heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer
from typing import Dict, List, Tuple, Optional
import re
import os


class ContentBERT(nn.Module):
    """
    DistilBERT-based Content Quality Analysis Model
    
    Multi-task outputs:
    - Argument strength (0-100)
    - Vocabulary sophistication (basic/intermediate/advanced)
    - Structure quality (0-100)
    - Engagement prediction (0-100)
    - Professionalism score (0-100)
    """
    
    VOCAB_LEVELS = ['basic', 'intermediate', 'advanced']
    
    def __init__(
        self,
        pretrained_model: str = 'distilbert-base-uncased',
        hidden_dim: int = 768,
        dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        super().__init__()
        
        # Load DistilBERT
        self.bert = DistilBertModel.from_pretrained(pretrained_model)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Pooling layer
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Task-specific heads
        self.argument_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.vocab_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # 3 classes
        )
        
        self.structure_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.engagement_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.professionalism_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: (batch, seq_len) tokenized input
            attention_mask: (batch, seq_len) attention mask
            
        Returns:
            Dictionary with all scores
        """
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        pooled = self.pooler(cls_output)
        
        # Multi-task outputs
        return {
            'argument_score': self.argument_head(pooled) * 100,
            'vocab_logits': self.vocab_head(pooled),
            'structure_score': self.structure_head(pooled) * 100,
            'engagement_score': self.engagement_head(pooled) * 100,
            'professionalism_score': self.professionalism_head(pooled) * 100
        }


class ContentAnalyzer:
    """
    High-level interface for content analysis
    Combines BERT model with rule-based analysis
    """
    
    # Professional vocabulary lists
    POWER_WORDS = {
        'significant', 'therefore', 'consequently', 'demonstrate', 'specifically',
        'critical', 'essential', 'methodology', 'conclusion', 'result', 'impact',
        'strategy', 'implementation', 'analysis', 'evidence', 'innovative',
        'comprehensive', 'fundamental', 'substantial', 'accordingly', 'moreover',
        'furthermore', 'nevertheless', 'consequently', 'subsequently', 'ultimately'
    }
    
    TRANSITION_PHRASES = {
        'on the other hand', 'in addition', 'furthermore', 'however', 'for example',
        'as a result', 'in conclusion', 'firstly', 'secondly', 'thirdly',
        'in contrast', 'similarly', 'as mentioned', 'to summarize', 'in particular',
        'more importantly', 'with regard to', 'in terms of', 'as we can see'
    }
    
    HEDGING_WORDS = {
        'maybe', 'perhaps', 'might', 'could', 'possibly', 'probably',
        'sort of', 'kind of', 'i think', 'i guess', 'i believe'
    }
    
    FILLER_WORDS = {
        'um', 'uh', 'like', 'literally', 'basically', 'actually',
        'you know', 'i mean', 'so yeah', 'right'
    }
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Initialize model
        self.model = ContentBERT().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded content model from {model_path}")
        
        self.model.eval()
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove non-speech artifacts
        text = re.sub(r'\[.*?\]', '', text)
        return text
    
    def analyze_rule_based(self, text: str) -> Dict:
        """Rule-based content analysis"""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        if word_count == 0:
            return {
                'word_count': 0,
                'power_word_ratio': 0,
                'transition_count': 0,
                'hedging_ratio': 0,
                'filler_ratio': 0,
                'avg_sentence_length': 0
            }
        
        # Power words
        power_count = sum(1 for w in words if w.strip('.,!?') in self.POWER_WORDS)
        power_ratio = power_count / word_count
        
        # Transitions
        transition_count = sum(1 for phrase in self.TRANSITION_PHRASES if phrase in text_lower)
        
        # Hedging
        hedging_count = sum(1 for phrase in self.HEDGING_WORDS if phrase in text_lower)
        hedging_ratio = hedging_count / max(word_count // 50, 1)  # Per 50 words
        
        # Fillers
        filler_count = sum(1 for w in words if w.strip('.,!?') in self.FILLER_WORDS)
        filler_ratio = filler_count / word_count
        
        # Sentence length
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = word_count / max(len(sentences), 1)
        
        return {
            'word_count': word_count,
            'power_word_count': power_count,
            'power_word_ratio': round(power_ratio * 100, 2),
            'transition_count': transition_count,
            'hedging_ratio': round(hedging_ratio, 2),
            'filler_ratio': round(filler_ratio * 100, 2),
            'avg_sentence_length': round(avg_sentence_length, 1),
            'sentence_count': len(sentences)
        }
    
    def detect_argument_structure(self, text: str) -> Dict:
        """Detect argumentative elements in text"""
        text_lower = text.lower()
        
        # Claim indicators
        claim_patterns = [
            r'\bwe (can|will|should|must)\b',
            r'\bthis (shows|demonstrates|proves|indicates)\b',
            r'\b(therefore|thus|hence|consequently)\b',
            r'\bour (approach|method|solution|strategy)\b'
        ]
        
        # Evidence indicators
        evidence_patterns = [
            r'\baccording to\b',
            r'\bresearch (shows|indicates|suggests)\b',
            r'\bstudies (show|indicate|suggest)\b',
            r'\b\d+(\.\d+)?%\b',  # Percentages
            r'\bin \d{4}\b',  # Years
            r'\bfor (example|instance)\b'
        ]
        
        claims = sum(len(re.findall(p, text_lower)) for p in claim_patterns)
        evidence = sum(len(re.findall(p, text_lower)) for p in evidence_patterns)
        
        return {
            'claim_indicators': claims,
            'evidence_indicators': evidence,
            'argument_ratio': round((claims + evidence) / max(len(text.split()) // 100, 1), 2)
        }
    
    def analyze(self, transcript: str) -> Dict:
        """
        Complete content analysis
        
        Args:
            transcript: presentation transcript
            
        Returns:
            Dictionary with all content metrics
        """
        # Preprocess
        text = self.preprocess_text(transcript)
        
        if not text:
            return {'error': 'Empty transcript'}
        
        # Rule-based analysis
        rule_results = self.analyze_rule_based(text)
        argument_results = self.detect_argument_structure(text)
        
        # Deep learning analysis
        # Truncate to model max length
        max_length = 512
        
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            dl_results = self.model(input_ids, attention_mask)
        
        vocab_probs = F.softmax(dl_results['vocab_logits'], dim=-1)
        vocab_level_idx = torch.argmax(vocab_probs, dim=-1).item()
        
        return {
            # Deep learning scores
            'argument_score': dl_results['argument_score'].item(),
            'vocabulary_level': ContentBERT.VOCAB_LEVELS[vocab_level_idx],
            'vocabulary_confidence': vocab_probs[0, vocab_level_idx].item(),
            'structure_score': dl_results['structure_score'].item(),
            'engagement_score': dl_results['engagement_score'].item(),
            'professionalism_score': dl_results['professionalism_score'].item(),
            
            # Rule-based metrics
            **rule_results,
            **argument_results
        }
    
    def calculate_score(self, analysis: Dict) -> Tuple[float, List[str]]:
        """Calculate presentation content score"""
        feedback = []
        
        if 'error' in analysis:
            return 50, ["Could not analyze content"]
        
        scores = []
        
        # Argument strength
        arg_score = analysis.get('argument_score', 50)
        scores.append(arg_score)
        
        # Vocabulary
        vocab_level = analysis.get('vocabulary_level', 'basic')
        if vocab_level == 'advanced':
            scores.append(90)
            feedback.append("Excellent use of sophisticated vocabulary")
        elif vocab_level == 'intermediate':
            scores.append(70)
        else:
            scores.append(50)
            feedback.append("Consider using more professional vocabulary")
        
        # Power words
        power_ratio = analysis.get('power_word_ratio', 0)
        if power_ratio > 3:
            feedback.append("Great use of impactful language")
        elif power_ratio < 1:
            feedback.append("Include more powerful, persuasive words")
        
        # Transitions
        trans_count = analysis.get('transition_count', 0)
        if trans_count >= 3:
            feedback.append("Good use of transitional phrases for flow")
        elif trans_count == 0:
            feedback.append("Add transitional phrases to improve flow")
        
        # Fillers
        filler_ratio = analysis.get('filler_ratio', 0)
        if filler_ratio > 2:
            feedback.append("Reduce filler words (um, like, basically)")
        
        # Hedging
        hedging_ratio = analysis.get('hedging_ratio', 0)
        if hedging_ratio > 2:
            feedback.append("Speak with more confidence - reduce hedging language")
        
        # Structure
        structure = analysis.get('structure_score', 50)
        scores.append(structure)
        
        # Professionalism
        prof = analysis.get('professionalism_score', 50)
        scores.append(prof)
        
        final_score = sum(scores) / len(scores)
        
        return round(final_score, 1), feedback


if __name__ == "__main__":
    # Test the model
    model = ContentBERT()
    print(f"Content BERT parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    sample_text = "Therefore, our innovative methodology demonstrates significant impact on the results."
    encoded = tokenizer(sample_text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
    
    outputs = model(encoded['input_ids'], encoded['attention_mask'])
    print("Content outputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v.mean().item():.2f}")
