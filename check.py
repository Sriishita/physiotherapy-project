import joblib
import os

def test_load_pickle(filepath):
    print(f"\n{'='*60}")
    print(f"Testing: {filepath}")
    print(f"{'='*60}")
    
    try:
        obj = joblib.load(filepath)
        print(f"✅ SUCCESS! Loaded object type: {type(obj)}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


test_files = [
    "models/arm_raise_model.pkl",
    "models/arm_raise_scaler.pkl",
    "models/leg_raise_model.pkl",
    "models/squat_model.pkl",
]

for f in test_files:
    test_load_pickle(f)