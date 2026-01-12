# 🚀 ModalX-AI: Multi-Modal Presentation Assessment System
**Daffodil AI Club - ModalX Challenge Submission**

## 📖 Project Overview
ModalX is an AI-powered grading assistant designed to help Faculty members evaluate student presentations objectively. By analyzing both **Audio (Verbal)** and **Visual (Non-Verbal)** cues, it generates a comprehensive "report card" with actionable feedback.

---

## ⚙️ Development Phases

### 🔹 Phase 1: Speech Analysis (Audio Intelligence)
*Located in `/Phase_1_Speech_Analysis`*
- **Goal:** Extract raw audio from video and analyze verbal delivery.
- **Tech Stack:** `OpenAI Whisper` (ASR), `Librosa` (Signal Processing).
- **Key Metrics:** Words Per Minute (WPM), Pitch/Tonal Variation, Filler Word Detection.

### 🔹 Phase 2: Motion & Visual Analysis (Behavioral AI)
*Located in `/Phase_2_Visual_Analysis`*
- **Goal:** Track body language and engagement.
- **Tech Stack:** `Google MediaPipe` (Holistic Model), `OpenCV`.
- **Key Metrics:** Eye Contact Consistency, Posture Stability, Hand Gesture tracking.

### 🔹 Phase 3: Full System Integration (The Product)
*Located in `/Phase_3_Full_System`*
- **Goal:** Merge Phase 1 & 2 into a faculty-facing SaaS tool.
- **Features:**
    - 📄 **Smart Report Generation:** Auto-generates PDFs with Grades (A+, B, etc.).
    - 🔗 **Google Drive Support:** Fetches videos directly via `gdown`.
    - 🙋 **Viva Question Generator:** Suggests questions based on speech content.
- **UI:** Built with `Streamlit` for a clean, responsive dashboard.

---

## 🛠️ Installation & Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
