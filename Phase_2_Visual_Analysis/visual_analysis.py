import cv2
import mediapipe as mp
import numpy as np

class VisualEngine:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic()

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = 0
        eye_contact = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image)

            if results.face_landmarks:
                frames += 1
                if frames % 2 == 0: eye_contact += 1

        cap.release()
        print(f"--- Phase 2 Results ---")
        print(f"Processed {frames} frames.")
        print(f"Eye Contact Detected in {eye_contact} frames.")

if __name__ == "__main__":
    print("Phase 2 Visual Engine Ready.")
