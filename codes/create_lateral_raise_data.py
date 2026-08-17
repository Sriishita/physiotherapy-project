import numpy as np
import os

os.makedirs("pose_data", exist_ok=True)

# same feature size as your existing data
feature_size = 99  # 33 landmarks * (x,y,z)

# create synthetic data
correct = np.random.normal(loc=0.5, scale=0.1, size=(120, feature_size))
incorrect = np.random.normal(loc=0.2, scale=0.2, size=(120, feature_size))

X = np.vstack([correct, incorrect])
y = np.array([1]*120 + [0]*120)

np.savez("pose_data/lateral_raise.npz", X=X, y=y)

print("✅ lateral_raise.npz created with dummy data")