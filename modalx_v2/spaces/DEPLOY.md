# Deploying ModalX v2 on Hugging Face Spaces (FREE GPU)

## Quick Deploy Steps

### 1. Create a Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Settings:
   - **Name:** `modalx-v2`
   - **SDK:** `Streamlit`
   - **Hardware:** `T4 small` (FREE GPU!)
   - **Visibility:** Public or Private

### 2. Upload Files

Upload these files from `modalx_v2/` to your Space:

```
app.py
backend.py
requirements.txt (use spaces/requirements.txt)
README.md (use spaces/README.md for HF format)
models/
  ├── __init__.py
  ├── transformer_emotion.py
  ├── action_unit_detector.py
  ├── gesture_stgcn.py
  ├── prosody_analyzer.py
  ├── content_bert.py
  └── slide_vit.py
weights/
  └── (your trained .pt files)
```

### 3. Using Git

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/modalx-v2
cd modalx-v2

# Copy files
cp -r /path/to/modalx_v2/* .
cp spaces/README.md .
cp spaces/requirements.txt .

# Push
git add .
git commit -m "Initial deploy"
git push
```

### 4. Wait for Build

The Space will build automatically. Takes ~5-10 minutes.

---

## Free GPU Limits

| Feature | Limit |
|---------|-------|
| GPU | T4 (16GB VRAM) |
| RAM | 16GB |
| Storage | 50GB |
| Timeout | 15 min idle |
| Cost | **FREE** |

---

## Troubleshooting

### Out of Memory
Use `whisper.load_model("tiny")` instead of "base"

### Slow Cold Start
First load takes ~2 minutes to download models

### Import Errors
Check that `protobuf>=3.20.0,<5.0.0` is in requirements.txt

---

## Your Space URL

After deployment:
```
https://huggingface.co/spaces/YOUR_USERNAME/modalx-v2
```

Share this link for the competition demo!
