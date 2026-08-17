"""
leg_raise_accuracy_fix.py
─────────────────────────
Run this ONCE to retrain the leg-raise model with:
  1. Better feature engineering (angles + ratios, not raw xyz)
  2. Ensemble model (RandomForest + GradientBoosting voting)
  3. Class-weight balancing
  4. Cross-val score printed so you can verify improvement

Usage:
    python leg_raise_accuracy_fix.py

It overwrites  models/leg_raise_model.pkl  and  models/leg_raise_scaler.pkl
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def angle_3pt_flat(kp, i_a, i_b, i_c):
    """Angle at joint b given flat keypoint array (x,y,z per landmark)."""
    def pt(i): return kp[i*3: i*3+2]          # x,y only
    va = pt(i_a) - pt(i_b)
    vc = pt(i_c) - pt(i_b)
    cos_a = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def extract_leg_raise_features(kp_flat):
    """
    kp_flat: 1-D array of shape (99,)  → 33 landmarks × (x, y, z)
    Returns a richer feature vector for leg raise.
    """
    feats = list(kp_flat)          # keep raw xyz as base

    # ── Joint angles ──────────────────────────────────────────────────────
    # Left:  hip(23) knee(25) ankle(27)
    # Right: hip(24) knee(26) ankle(28)
    l_knee_ang = angle_3pt_flat(kp_flat, 23, 25, 27)
    r_knee_ang = angle_3pt_flat(kp_flat, 24, 26, 28)

    # Hip flexion: shoulder(11)→hip(23)→knee(25)
    l_hip_ang  = angle_3pt_flat(kp_flat, 11, 23, 25)
    r_hip_ang  = angle_3pt_flat(kp_flat, 12, 24, 26)

    feats += [l_knee_ang, r_knee_ang, l_hip_ang, r_hip_ang]

    # ── Vertical ratios (ankle y relative to hip y) ───────────────────────
    # MediaPipe: y=0 top, y=1 bottom → raised leg has smaller y than hip
    l_ankle_y = kp_flat[27*3 + 1]
    r_ankle_y = kp_flat[28*3 + 1]
    l_hip_y   = kp_flat[23*3 + 1]
    r_hip_y   = kp_flat[24*3 + 1]

    l_raise_ratio = l_hip_y - l_ankle_y   # positive = raised
    r_raise_ratio = r_hip_y - r_ankle_y
    feats += [l_raise_ratio, r_raise_ratio]

    # ── Leg straightness (deviation from 180°) ────────────────────────────
    l_bend = abs(180 - l_knee_ang)
    r_bend = abs(180 - r_knee_ang)
    feats += [l_bend, r_bend]

    # ── Symmetry signals ─────────────────────────────────────────────────
    feats.append(abs(l_knee_ang - r_knee_ang))   # knee angle diff
    feats.append(abs(l_raise_ratio - r_raise_ratio))  # raise height diff

    return np.array(feats, dtype=np.float32)


def build_feature_matrix(raw_X):
    """raw_X: (N, 99) array of raw keypoints → (N, n_features)"""
    return np.array([extract_leg_raise_features(row) for row in raw_X])


# ─────────────────────────────────────────────────────────────────────────────
# LOAD EXISTING DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    data_path = Path("pose_data/augmented/leg_raise.npz")
    if not data_path.exists():
        # Try non-augmented fallback
        data_path = Path("pose_data/leg_raise.npz")
    if not data_path.exists():
        raise FileNotFoundError(
            "Cannot find leg_raise.npz in pose_data/ or pose_data/augmented/\n"
            "Run your data collection scripts first."
        )
    data = np.load(data_path)
    X, y = data["X"], data["y"]
    print(f"[DATA] Loaded {len(X)} samples  |  class 0: {(y==0).sum()}  class 1: {(y==1).sum()}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train():
    X_raw, y = load_data()

    print("[FEAT] Engineering features ...")
    X = build_feature_matrix(X_raw)
    print(f"[FEAT] Feature vector size: {X.shape[1]}")

    # ── Scaler ────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # ── Ensemble ──────────────────────────────────────────────────────────
    rf  = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",   # handles imbalance automatically
        random_state=42,
        n_jobs=-1,
    )
    gb  = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )
    model = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",             # average probabilities
        weights=[1, 1],
    )

    # ── Cross-validation BEFORE saving ────────────────────────────────────
    print("[CV]  Running 5-fold stratified CV ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_sc, y, cv=cv, scoring="accuracy")
    print(f"[CV]  Accuracy: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")

    f1_scores = cross_val_score(model, X_sc, y, cv=cv, scoring="f1")
    print(f"[CV]  F1 score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    # ── Final fit on ALL data ─────────────────────────────────────────────
    print("[FIT] Training on full dataset ...")
    model.fit(X_sc, y)

    # ── Save ──────────────────────────────────────────────────────────────
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model,  "models/leg_raise_model.pkl")
    joblib.dump(scaler, "models/leg_raise_scaler.pkl")

    # Also save a feature-extractor flag so rehab_system knows to use it
    joblib.dump(True, "models/leg_raise_use_features.pkl")

    print("\n[DONE] Saved:")
    print("         models/leg_raise_model.pkl")
    print("         models/leg_raise_scaler.pkl")
    print("         models/leg_raise_use_features.pkl")
    print("\nNow update rehabilitation_system.py — see the patch below.\n")
    _print_patch()


# ─────────────────────────────────────────────────────────────────────────────
# PATCH INSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _print_patch():
    patch = """
── PATCH for rehabilitation_system.py ──────────────────────────────────────

1.  Add this import near the top:
        from leg_raise_accuracy_fix import extract_leg_raise_features

2.  In RehabSystem.__init__, load the flag:
        import os
        self.use_leg_features = os.path.exists(f"models/{exercise}_use_features.pkl")

3.  Replace the ML block inside run() with:

        kp_input = kp  # raw (99,) array

        if self.exercise == "leg_raise" and self.use_leg_features:
            kp_input = extract_leg_raise_features(kp)

        X = self.scaler.transform(kp_input.reshape(1, -1))
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0]
        conf = float(prob[pred])

─────────────────────────────────────────────────────────────────────────────
"""
    print(patch)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()