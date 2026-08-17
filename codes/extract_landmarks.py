import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ─────────────────────────────────────────────────────────────
# INIT MEDIAPIPE (NEW API)
# ─────────────────────────────────────────────────────────────
base = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")

options = vision.PoseLandmarkerOptions(
    base_options=base,
    running_mode=vision.RunningMode.IMAGE
)

pose = vision.PoseLandmarker.create_from_options(options)


# ─────────────────────────────────────────────────────────────
# EXTRACT LANDMARKS FROM FRAMES
# ─────────────────────────────────────────────────────────────
def extract_landmarks_from_frames(frames_dir):
    X = []
    y = []

    for label, folder in enumerate(["incorrect", "correct"]):
        path = os.path.join(frames_dir, folder)

        if not os.path.exists(path):
            print(f"[WARNING] Missing folder: {path}")
            continue

        files = sorted(os.listdir(path))

        for file in files:
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to mediapipe format
            img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            result = pose.detect(img_mp)

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]

                kp = np.array([[p.x, p.y, p.z] for p in lm]).flatten()

                X.append(kp)
                y.append(label)

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────
# PROCESS EACH EXERCISE
# ─────────────────────────────────────────────────────────────
def process_exercise(exercise):
    print(f"\n[INFO] Processing: {exercise}")

    frames_dir = f"frames/{exercise}"

    X, y = extract_landmarks_from_frames(frames_dir)

    if len(X) == 0:
        print(f"[ERROR] No data for {exercise}")
        return

    os.makedirs("pose_data", exist_ok=True)

    save_path = f"pose_data/{exercise}.npz"
    np.savez(save_path, X=X, y=y)

    print(f"[SAVED] {exercise} -> {save_path}")
    print(f"Samples: {len(X)}")
    print(f"Correct: {np.sum(y==1)} | Incorrect: {np.sum(y==0)}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    exercises = ["arm_raise", "leg_raise"]

    for ex in exercises:
        process_exercise(ex)

    print("\n✅ Landmark extraction complete!")