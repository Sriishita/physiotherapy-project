import numpy as np
import os

def view_npz_file(filepath):
    """View contents of a .npz file"""
    print(f"\n{'='*60}")
    print(f"File: {filepath}")
    print(f"{'='*60}")
    
    data = np.load(filepath)
    
    print(f"\nKeys in file: {list(data.keys())}")
    
    for key in data.keys():
        arr = data[key]
        print(f"\n{key}:")
        print(f"  Shape: {arr.shape}")
        print(f"  Data type: {arr.dtype}")
        
        if key == 'X':
            print(f"  Features per sample: {arr.shape[1]}")
            print(f"  First sample (first 10 values): {arr[0][:10]}")
        elif key == 'y':
            unique, counts = np.unique(arr, return_counts=True)
            print(f"  Labels: {dict(zip(unique, counts))}")
            print(f"    - Correct (1): {np.sum(arr == 1)}")
            print(f"    - Incorrect (0): {np.sum(arr == 0)}")

if __name__ == "__main__":
    pose_data_dir = "pose_data/augmented"
    
    if not os.path.exists(pose_data_dir):
        print(f"[ERROR] Directory '{pose_data_dir}' not found!")
        exit()
    
    npz_files = [f for f in os.listdir(pose_data_dir) if f.endswith('.npz')]
    
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{pose_data_dir}'!")
        exit()
    
    print(f"Found {len(npz_files)} .npz files")
    
    for npz_file in sorted(npz_files):
        filepath = os.path.join(pose_data_dir, npz_file)
        view_npz_file(filepath)
    
    print(f"\n{'='*60}\n")
