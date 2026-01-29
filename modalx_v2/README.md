# ModalX v2.0 - Deep Learning Presentation Grader

**Competition:** ModalX-AI Challenge (Daffodil International University)  
**Team:** NL Circuits (Muntasir Islam, Nazmus Sakib)

---

## 🧠 What's New in v2.0

This version introduces **6 state-of-the-art deep learning models** for comprehensive presentation analysis:

| Model | Architecture | Purpose |
|-------|--------------|---------|
| **Emotion Analyzer** | Transformer + Multi-Head Attention | Speech emotion recognition |
| **AU Detector** | ResNet-50 + LSTM | Facial Action Unit detection |
| **Gesture Analyzer** | ST-GCN (Graph Neural Network) | Body language classification |
| **Prosody Analyzer** | CNN-BiLSTM | Speech quality metrics |
| **Content Scorer** | DistilBERT | Transcript quality assessment |
| **Slide Analyzer** | Vision Transformer (ViT) | Slide design grading |

---

## 📁 Project Structure

```
modalx_v2/
├── app.py                  # Streamlit dashboard
├── backend.py              # Unified analysis engine
├── models/                 # Deep learning models
│   ├── transformer_emotion.py
│   ├── action_unit_detector.py
│   ├── gesture_stgcn.py
│   ├── prosody_analyzer.py
│   ├── content_bert.py
│   └── slide_vit.py
├── training/               # Google Colab notebooks
│   ├── train_emotion_colab.ipynb
│   ├── train_gesture_colab.ipynb
│   └── train_content_colab.ipynb
├── weights/                # Trained model weights
├── Dockerfile              # DigitalOcean deployment
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Quick Start

### Local Development

```bash
cd modalx_v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### Docker Deployment (DigitalOcean)

```bash
# Build and run
docker-compose up -d

# Or single command
docker build -t modalx-v2 . && docker run -p 8501:8501 modalx-v2
```

---

## 🎓 Training Models (Google Colab)

1. Open notebook in `training/` folder
2. Upload to Google Colab
3. Enable GPU runtime (Runtime → Change runtime type → T4 GPU)
4. Run all cells
5. Download trained weights to `weights/` directory

### Datasets Used

| Model | Dataset | Source |
|-------|---------|--------|
| Emotion | RAVDESS, TESS | Kaggle |
| Gesture | Custom collected | MediaPipe skeletons |
| Content | TED Talk transcripts | Custom annotations |
| Slides | Presentation slides | Custom collected |

---

## 📊 Scoring System

**Weighted Final Score (100 points):**

| Component | Weight | Models Used |
|-----------|--------|-------------|
| Audio Quality | 20% | Prosody Analyzer |
| Visual Behavior | 20% | AU Detector + Gesture ST-GCN |
| Emotion Intelligence | 20% | Transformer Emotion |
| Content Quality | 20% | DistilBERT Content |
| Slide Design | 20% | ViT Slide Analyzer |

---

## 🛠️ Tech Stack

- **Framework:** PyTorch, Transformers, timm
- **Speech:** OpenAI Whisper, torchaudio, librosa
- **Vision:** MediaPipe, OpenCV, pytesseract
- **NLP:** DistilBERT, HuggingFace Transformers
- **Frontend:** Streamlit, Plotly
- **Deployment:** Docker, DigitalOcean

---

## 📝 API Usage

```python
from backend import ModalXSystemV2

# Initialize system
system = ModalXSystemV2(weights_dir="weights")

# Analyze presentation
results = system.analyze(
    video_path="presentation.mp4",
    student_name="John Doe",
    student_id="123456",
    is_url=False
)

print(f"Score: {results['score']}/100")
print(f"Feedback: {results['feedback']}")
```

---

## 🏆 Competition Features

1. **No External APIs** - All models run locally
2. **Deep Learning Focus** - 6 neural network architectures
3. **Multi-Modal Analysis** - Audio + Visual + Text
4. **PDF Reports** - Professional grading documents
5. **Production Ready** - Docker deployment for DigitalOcean

---

## 📄 License

MIT License - Free for educational and competition use.

---

**Built with ❤️ for ModalX-AI Challenge @ DIU**
