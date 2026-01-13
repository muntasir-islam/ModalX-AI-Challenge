# 🎓 ModalX: The Smart Faculty Grading Assistant
**AI-Powered Presentation Assessment System**

> 🔴 **Live Demo:** [ml.muntasirislam.com/modalx](https://ml.muntasirislam.com/modalx)

---

## 💡 The Problem We Are Solving
Faculty members and lecturers face a massive challenge: **Grading Fatigue.**

Imagine sitting through 50 student presentations in a single day. By the 40th student, it becomes incredibly difficult to remain objective, track every pause, or measure eye contact accurately. Grading becomes subjective, and providing detailed, personalized feedback for every single student is nearly impossible due to time constraints.

**ModalX is built to be the Lecturer's "Helping Hand".**
It automates the tedious parts of grading (measuring pace, volume, slide density, and posture) so the faculty member can focus on what matters most: the **content and the idea**.

---

## ⚙️ How It Helps the Faculty
We designed this system to act as an objective, "Responsible AI" grader that follows a standard rubric.

### 1. Automated Rubric & Evidence
Instead of just giving a grade like "B+", ModalX generates a **Detailed PDF Report** for every student. This serves as:
* **Proof of Assessment:** A physical record of why a student received a specific grade.
* **Objectivity:** The AI doesn't get tired. It grades the first student and the last student with the exact same criteria.

### 2. The "Context-Aware" Grading Engine
Lecturers encounter different types of presentations. A "One-size-fits-all" AI fails here. We built a **"Smart Pivot"** engine that adapts:

* **Scenario A: The Speech.** If the student is on camera, the AI uses `MediaPipe` to grade **Eye Contact** and **Posture Stability**.
* **Scenario B: The Screen Share.** If the student is showing slides (and their face is hidden), the AI automatically switches to **OCR Mode**. It reads the slides using `Tesseract` to check for **Readability** and **Text Density** (avoiding "Death by PowerPoint").

### 3. Deep Audio Forensics
We go beyond simple transcription.
* **Confidence scoring:** Using RMS Energy to detect mumbling vs. projection.
* **Hesitation Analysis:** Distinguishing between a "Dramatic Pause" (good) and "Nervous Stuttering" (needs improvement).

---

## 📂 Project Architecture (Phase by Phase)
This repository documents our development journey for the ModalX AI Challenge:

* **`Phase_1_Speech_Analysis/`**: The core audio processing logic. We built the "Prosody Engine" here to measure pitch and pacing.
* **`Phase_2_Visual_Analysis/`**: The computer vision layer. This contains the logic for the "Smart Pivot" (Face vs. Slide detection).
* **`Phase_3_Full_System/`**: The production-ready web application hosted on the live server.

---

## 🛠️ Deployment & Usage
The project is currently hosted live, but you can run a local instance for testing.

**1. Clone & Install**
```bash
git clone https://github.com/muntasir-islam/ModalX-Repo.git
pip install -r requirements.txt
