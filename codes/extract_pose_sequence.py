import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_pose_keypoints(image_path):
    """
    Extract 33 pose landmarks (x, y, z) from a single image.
    Returns a numpy array of length 99 (33 landmarks × 3).
    """
    
    # Create PoseLandmarker
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        # Read image
        image = mp.Image.create_from_file(image_path)
        
        # Detect pose
        detection_result = landmarker.detect(image)
        
        if detection_result.pose_landmarks:
            keypoints = []
            for landmark in detection_result.pose_landmarks[0]:
                keypoints.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])
            return np.array(keypoints)
        
        return None


def process_exercise_frames(
    exercise_name,
    frames_base_dir="frames",
    output_dir="pose_data"
):
    """
    Process all frames for one exercise and save pose data as .npz file.
    """
    
    print(f"\n[INFO] Processing exercise: {exercise_name}")
    
    exercise_path = os.path.join(frames_base_dir, exercise_name)
    if not os.path.exists(exercise_path):
        print(f"[ERROR] Exercise folder not found: {exercise_path}")
        return
    
    X = []  # Pose features
    y = []  # Labels (1 = correct, 0 = incorrect)
    
    for label_name, label_value in [("correct", 1), ("incorrect", 0)]:
        label_dir = os.path.join(exercise_path, label_name)
        
        if not os.path.exists(label_dir):
            print(f"[WARNING] Label folder not found: {label_dir}")
            continue
        
        frame_files = sorted(
            f for f in os.listdir(label_dir) if f.endswith(".jpg")
        )
        
        print(f"[INFO] Found {len(frame_files)} frames in {label_name}")
        
        for frame_file in frame_files:
            frame_path = os.path.join(label_dir, frame_file)
            keypoints = extract_pose_keypoints(frame_path)
            
            if keypoints is not None:
                X.append(keypoints)
                y.append(label_value)
    
    if len(X) == 0:
        print(f"[ERROR] No valid pose data extracted for {exercise_name}")
        return
    
    X = np.array(X)
    y = np.array(y)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{exercise_name}.npz")
    np.savez(output_file, X=X, y=y)
    
    print(f"[SUCCESS] Saved {len(X)} samples to {output_file}")
    print(f"  - Shape: X={X.shape}, y={y.shape}")
    print(f"  - Correct: {np.sum(y == 1)}, Incorrect: {np.sum(y == 0)}")


if __name__ == "__main__":
    
    frames_base = "frames"
    pose_output = "pose_data"
    
    print("[INFO] Starting pose extraction...")
    print(f"[INFO] Reading frames from: {os.path.abspath(frames_base)}")
    
    if not os.path.exists(frames_base):
        print(f"[ERROR] Frames directory not found: {frames_base}")
        exit()
    
    exercises = [
        d for d in os.listdir(frames_base)
        if os.path.isdir(os.path.join(frames_base, d))
    ]
    
    if not exercises:
        print("[ERROR] No exercise folders found!")
        exit()
    
    print(f"[INFO] Found exercises: {exercises}")
    
    for exercise in exercises:
        process_exercise_frames(exercise, frames_base, pose_output)
    
    print("\n[INFO] Pose extraction complete!")