# ModalX-AI Challenge: Presentation Assessment System

* **Team Name:** NL Circuits
* **Team Members:** Muntasir Islam, Nazmus Sakib
* **Student IDs:** 0242220005131010, 0242220005131017
* **Event:** ModalX-AI Challenge (Daffodil International University)

---

## Project Overview

ModalX is an automated grading system for student presentations. It replaces subjective human grading with data-driven metrics. The system analyzes three key areas:
1.  **Audio:** Speaking pace, volume, and pitch variation.
2.  **Visual:** Eye contact, body posture, and slide detection (analyzing both slide content and visuals).
3.  **Emotion:** Vocal tone and confidence levels.

This repository is organized into three folders corresponding to the competition phases.

---

## Repository Structure

### 1. Phase_1_Speech_Analysis
This folder contains the logic for processing audio.
* **Focus:** Speech Clarity, Confidence, and Delivery.
* **Key Metrics:**
    * **Words Per Minute (WPM):** measures speaking speed.
    * **Pitch Variation:** detects if the voice is monotone.
    * **Pause Ratio:** measures hesitation.
    * **Emotion Analysis:** detects emotions and measures confidence scales.
    * **Emotion detection Model Training:** dataset from [text](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

* **Tech Stack:** OpenAI Whisper, Librosa, TensorFlow.

### 2. Phase_2_Visual_Analysis
This folder contains the logic for computer vision (Face, Body movement analysis, and Slide content/visual analysis).
* **Focus:** Body Language, Engagement, and Slide Contents.
* **Key Metrics:**
    * **Eye Contact:** tracks if the speaker is looking at the camera.
    * **Posture Stability:** checks for slouching, movement, hand movements, and gestures.
    * **Slide Detection:** differentiates between a person speaking and screen-shared slides; measures slide content density and visual content ratios.
* **Tech Stack:** MediaPipe Holistic, OpenCV.

### 3. Phase_3_Full_System
This is the final integrated application. It combines Phase 1 and Phase 2, adds the Emotion Engine, and generates the final report.
* **Focus:** System Integration and Reporting.
* **Features:**
    * **Emotion Engine:** A CNN model that detects happy, nervous, or neutral tones.
    * **Content Analysis:** Scans the transcript for professional vocabulary.
    * **PDF Report:** Automatically generates a scorecard with a final grade.
* **Main File:** `app.py` (Streamlit Dashboard). `backend.py` (All algorithms and processing)

---

## Installation Guide

Follow these steps to set up the project on your local machine.

**1. Clone the Repository**
```bash
git clone [https://github.com/muntasir-islam/ModalX-AI-Challenge.git](https://github.com/muntasir-islam/ModalX-AI-Challenge.git)
cd ModalX-AI-Challenge