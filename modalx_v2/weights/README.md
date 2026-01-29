# ModalX v2 Model Weights Directory

This directory stores trained model weights for deployment.

## Required Weight Files

After training on Google Colab, place these files here:

| File | Model | Size (approx) |
|------|-------|---------------|
| `emotion_transformer.pt` | Transformer Emotion | ~15MB |
| `au_detector.pt` | ResNet-50 AU Detector | ~100MB |
| `gesture_stgcn.pt` | ST-GCN Gesture | ~5MB |
| `prosody_model.pt` | CNN-BiLSTM Prosody | ~10MB |
| `content_bert.pt` | DistilBERT Content | ~250MB |
| `slide_vit.pt` | ViT Slide Analyzer | ~350MB |

## Training Notebooks

Train models using Google Colab notebooks in `training/`:
- `train_emotion_colab.ipynb`
- `train_gesture_colab.ipynb`
- `train_content_colab.ipynb`

## Download Pretrained (Optional)

```bash
# Create weights directory
mkdir -p weights

# Models will be downloaded from our releases
# (Add actual download links after training)
```

## Notes

- The app will work without pretrained weights (using random initialization)
- For best results, train on competition-specific data
- GPU (CUDA) required for efficient training
