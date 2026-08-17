import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(exercise):
    """Evaluate a trained posture classification model."""
    print(f"\n{'='*60}")
    print(f"Evaluating Model for: {exercise.upper()}")
    print(f"{'='*60}")

    # File paths
    model_path = f"models/{exercise}_model.pkl"
    scaler_path = f"models/{exercise}_scaler.pkl"
    test_data_path = f"pose_data/augmented/{exercise}.npz"

    # --- Check files exist ---
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None
    if not os.path.exists(scaler_path):
        print(f"[ERROR] Scaler file not found: {scaler_path}")
        return None
    if not os.path.exists(test_data_path):
        print(f"[ERROR] Test data file not found: {test_data_path}")
        return None

    try:
        # --- Load model and data ---
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        data = np.load(test_data_path)

        # Handle both formats safely
        if 'X' in data and 'y' in data:
            X, y = data["X"], data["y"]
        else:
            keys = data.files
            X, y = data[keys[0]], data[keys[1]]

        print(f"[INFO] Loaded data: {X.shape}")

        # --- Scale & Predict ---
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)

        # --- Metrics ---
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)

        print(f"\nAccuracy: {acc*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y, preds, target_names=["Incorrect", "Correct"]))
        print("Confusion Matrix:")
        print(cm)

        # --- Save Confusion Matrix ---
        os.makedirs("results", exist_ok=True)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Incorrect", "Correct"])
        disp.plot(values_format="d")  # ❌ removed cmap (safer per rules)
        
        plt.title(f"{exercise.capitalize()} - Confusion Matrix ({acc*100:.2f}% Accuracy)")
        save_path = f"results/{exercise}_confusion_matrix.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[SAVED] Confusion matrix: {save_path}")

        return acc * 100

    except Exception as e:
        print(f"[ERROR] Failed during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🚀 Starting Model Evaluation")

    exercises = ["arm_raise", "leg_raise", "squat"]
    results = {}

    for ex in exercises:
        acc = evaluate_model(ex)
        if acc is not None:
            results[ex] = acc

    print("\n=== Summary of Accuracies ===")
    for ex, acc in results.items():
        print(f"{ex.capitalize()}: {acc:.2f}%")

    if results:
        print(f"\nAverage Accuracy: {np.mean(list(results.values())):.2f}%")