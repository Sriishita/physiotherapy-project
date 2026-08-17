import numpy as np
import os

def augment_pose_sequence(seq):
    """Apply augmentation to a single pose sequence."""
    seq = seq.copy()
    
    # Add random noise
    noise = np.random.normal(0, 0.01, seq.shape)
    seq_noisy = seq + noise
    
    # Random scaling
    scale = np.random.uniform(0.95, 1.05)
    seq_scaled = seq_noisy * scale
    
    # Random shift for each (x, y, z) triplet
    shift = np.random.uniform(-0.02, 0.02, (1, 3))
    for i in range(0, seq_scaled.shape[0], 3):
        seq_scaled[i:i+3] += shift.flatten()
    
    return seq_scaled

def augment_dataset(input_path, output_path, n_aug=3):
    """Augment a single .npz file and save the augmented version."""
    print(f"[INFO] Loading {input_path}")
    
    # Check if file exists and has content
    if not os.path.exists(input_path):
        print(f"[ERROR] File not found: {input_path}")
        return
    
    if os.path.getsize(input_path) == 0:
        print(f"[ERROR] File is empty: {input_path}")
        return
    
    # Load the data
    data = np.load(input_path)
    keys = data.files
    
    # Handle both cases: saved as 'X','y' or unnamed arrays
    if 'X' in keys and 'y' in keys:
        X, y = data['X'], data['y']
    else:
        X, y = data[keys[0]], data[keys[1]]
    
    print(f"[INFO] Original data shape: X={X.shape}, y={y.shape}")
    
    # Create augmented samples
    X_aug, y_aug = [], []
    for i in range(len(X)):
        for _ in range(n_aug):
            X_aug.append(augment_pose_sequence(X[i]))
            y_aug.append(y[i])
    
    # Combine original and augmented data
    X_total = np.concatenate([X, np.array(X_aug)])
    y_total = np.concatenate([y, np.array(y_aug)])
    
    # Save augmented dataset
    np.savez(output_path, X=X_total, y=y_total)
    print(f"[SUCCESS] Augmented data saved: {X_total.shape[0]} samples → {output_path}")
    print(f"  - Original: {len(X)} samples")
    print(f"  - Augmented: {len(X_aug)} samples")
    print(f"  - Total: {len(X_total)} samples\n")

if __name__ == "__main__":
    base_dir = "pose_data"
    output_dir = os.path.join(base_dir, "augmented")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("[INFO] Starting data augmentation...")
    print(f"[INFO] Working directory: {os.getcwd()}")
    print(f"[INFO] Input directory: {os.path.abspath(base_dir)}")
    print(f"[INFO] Output directory: {os.path.abspath(output_dir)}\n")
    
    # Find all .npz files (excluding already augmented ones)
    files = [f for f in os.listdir(base_dir) 
             if f.endswith(".npz") and os.path.isfile(os.path.join(base_dir, f))]
    
    if not files:
        print(f"[ERROR] No .npz files found in {base_dir}/")
        print("[INFO] Make sure you've run the pose extraction script first!")
        exit()
    
    print(f"[INFO] Found {len(files)} file(s) to augment: {files}\n")
    
    # Augment each file
    for file in files:
        input_path = os.path.join(base_dir, file)
        output_path = os.path.join(output_dir, file)
        augment_dataset(input_path, output_path, n_aug=5)
    
    print("[INFO] Data augmentation complete!")