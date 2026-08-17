import os
import cv2

def extract_frames(video_path, output_dir, label, skip_rate=5):
    """Extract frames from a single video and save to output directory."""
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved = 0
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    else:
        print(f"[INFO] Successfully opened: {video_path}")
        print(f"[INFO] Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % skip_rate == 0:
            filename = os.path.join(output_dir, label, f"{label}_{saved:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1
        
        frame_id += 1
    
    cap.release()
    print(f"[INFO] {saved} frames extracted from {video_path}")

if __name__ == "__main__":
    base_videos = "videos"
    base_output = "frames"
    
    print(f"[DEBUG] Looking for videos in: {os.path.abspath(base_videos)}")
    
    # Check if base_videos directory exists
    if not os.path.exists(base_videos):
        print(f"[ERROR] Directory '{base_videos}' does not exist!")
        exit()
    
    # Get all exercises (arm_raise, leg_raise, squat)
    exercises = [d for d in os.listdir(base_videos) if os.path.isdir(os.path.join(base_videos, d))]
    print(f"[DEBUG] Found exercises: {exercises}")
    
    if not exercises:
        print("[ERROR] No exercise folders found!")
        exit()
    
    for ex in exercises:
        print(f"\n[DEBUG] Processing exercise: {ex}")
        ex_path = os.path.join(base_videos, ex)
        out_path = os.path.join(base_output, ex)
        
        # Loop through both correct and incorrect videos
        for label in ["correct", "incorrect"]:
            video_file = os.path.join(ex_path, f"{label}.mp4")
            print(f"[DEBUG] Checking for: {video_file}")
            
            if os.path.exists(video_file):
                print(f"[DEBUG] Found video, extracting frames...")
                extract_frames(video_file, out_path, label)
            else:
                print(f"[WARNING] Video not found: {video_file}")