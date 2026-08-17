import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_augmented_data(augmented_dir="pose_data"):
    """Load all augmented .npz files and combine them."""
    print(f"[INFO] Loading original data from: {augmented_dir}")
    
    X_all = []
    y_all = []
    exercise_labels = []
    
    files = [f for f in os.listdir(augmented_dir) if f.endswith('.npz')]
    
    if not files:
        print(f"[ERROR] No .npz files found in {augmented_dir}")
        return None, None, None
    
    for i, file in enumerate(sorted(files)):
        file_path = os.path.join(augmented_dir, file)
        data = np.load(file_path)
        
        X = data['X']
        y = data['y']
        
        exercise_name = file.replace('.npz', '')
        print(f"[INFO] Loaded {exercise_name}: {len(X)} samples")
        
        X_all.append(X)
        y_all.append(y)
        
        exercise_labels.extend([i] * len(X))
    
    X_combined = np.vstack(X_all)
    y_combined = np.concatenate(y_all)
    exercise_labels = np.array(exercise_labels)
    
    print(f"\n[INFO] Total combined data: {len(X_combined)} samples")
    print(f"  - Correct poses: {np.sum(y_combined == 1)}")
    print(f"  - Incorrect poses: {np.sum(y_combined == 0)}")
    
    return X_combined, y_combined, exercise_labels


def train_model(X, y, exercise_labels):
    """Train a Random Forest classifier."""
    print("\n[INFO] Starting model training...")
    
    X_train, X_test, y_train, y_test, ex_train, ex_test = train_test_split(
        X, y, exercise_labels, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Training set: {len(X_train)} samples")
    print(f"[INFO] Test set: {len(X_test)} samples")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # keep your noise logic (unchanged)
    X_train_scaled = X_train_scaled + np.random.normal(0, 0.8, X_train_scaled.shape)
    X_test_scaled = X_test_scaled + np.random.normal(0, 0.8, X_test_scaled.shape)
    
    print("\n[INFO] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    # ========== PROFESSIONAL VISUALIZATION FOR COMBINED MODEL ==========
    # Confusion Matrix Graph
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Incorrect", "Correct"],
                yticklabels=["Incorrect", "Correct"],
                linewidths=2, linecolor='white')
    
    plt.xlabel("Predicted", fontsize=12, fontweight='bold')
    plt.ylabel("Actual", fontsize=12, fontweight='bold')
    plt.title("Confusion Matrix - Combined Model", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    plt.savefig(os.path.join(models_dir, "combined_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Professional Accuracy Bar Graph with percentage label
    accuracy = accuracy_score(y_test, y_pred)
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(["Accuracy"], [accuracy], alpha=0.8, color='#2E86AB', edgecolor='#1B4965', linewidth=2)
    
    plt.ylim(0, 1)
    plt.title("Combined Model Accuracy", fontsize=16, fontweight='bold')
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    
    # Add percentage label on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                 f"{accuracy*100:.2f}%", ha='center', fontsize=12, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(os.path.join(models_dir, "combined_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.show()
    # ========== END VISUALIZATION ==========
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    print(f"\n[RESULT] Test Accuracy: {accuracy*100:.2f}%")
    
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Incorrect', 'Correct']))
    
    print("\n[RESULT] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    joblib.dump(model, os.path.join(models_dir, "posture_classifier.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    print(f"\n[SUCCESS] Model saved to: models/posture_classifier.pkl")
    print(f"[SUCCESS] Scaler saved to: models/scaler.pkl")
    
    return model, scaler, accuracy


def train_exercise_specific_models(augmented_dir="pose_data"):
    """Train separate models for each exercise type."""
    print("\n" + "="*50)
    print("TRAINING EXERCISE-SPECIFIC MODELS")
    print("="*50)
    
    exercises = ['arm_raise', 'leg_raise', 'squat', 'lateral_raise']
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    results = {}
    
    for exercise in exercises:
        file_path = os.path.join(augmented_dir, f"{exercise}.npz")
        
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}")
            continue
        
        print(f"\n[INFO] Training model for: {exercise}")
        data = np.load(file_path)
        X = data['X']
        y = data['y']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = X_train_scaled + np.random.normal(0, 0.8, X_train_scaled.shape)
        X_test_scaled = X_test_scaled + np.random.normal(0, 0.8, X_test_scaled.shape)
        
        model = RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # ========== PROFESSIONAL VISUALIZATION FOR EACH EXERCISE ==========
        # Confusion Matrix Graph
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Incorrect", "Correct"],
                    yticklabels=["Incorrect", "Correct"],
                    linewidths=2, linecolor='white')
        
        plt.xlabel("Predicted", fontsize=12, fontweight='bold')
        plt.ylabel("Actual", fontsize=12, fontweight='bold')
        plt.title(f"Confusion Matrix - {exercise.replace('_', ' ').title()}", 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(os.path.join(models_dir, f"{exercise}_confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Professional Horizontal Accuracy Bar (cleaner for presentations)
        plt.figure(figsize=(7, 3))
        bars = plt.barh(["Accuracy"], [accuracy], color='#2E86AB', edgecolor='#1B4965', linewidth=2)
        plt.xlim(0, 1)
        
        plt.title(f"{exercise.replace('_', ' ').title()} Model Accuracy: {accuracy*100:.2f}%", 
                  fontsize=14, fontweight='bold')
        plt.xlabel("Score", fontsize=12, fontweight='bold')
        
        # Add percentage label
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                     f"{accuracy*100:.2f}%", va='center', fontsize=11, fontweight='bold')
        
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig(os.path.join(models_dir, f"{exercise}_accuracy.png"), dpi=300, bbox_inches='tight')
        plt.show()
        # ========== END VISUALIZATION ==========
        
        print(f"[RESULT] {exercise} - Accuracy: {accuracy*100:.2f}%")
        
        joblib.dump(model, os.path.join(models_dir, f"{exercise}_model.pkl"))
        joblib.dump(scaler, os.path.join(models_dir, f"{exercise}_scaler.pkl"))
        
        print(f"[SUCCESS] Saved: models/{exercise}_model.pkl")
        results[exercise] = accuracy
    
    # ========== PROFESSIONAL COMPARISON BAR CHART ==========
    if results:
        plt.figure(figsize=(10, 6))
        exercises_list = [ex.replace('_', ' ').title() for ex in results.keys()]
        accuracies = list(results.values())
        
        bars = plt.bar(exercises_list, accuracies, alpha=0.8, 
                       color='#2E86AB', edgecolor='#1B4965', linewidth=2)
        plt.ylim(0, 1)
        
        plt.title("Model Accuracy Comparison by Exercise Type", fontsize=16, fontweight='bold')
        plt.ylabel("Accuracy Score", fontsize=12, fontweight='bold')
        plt.xlabel("Exercise Type", fontsize=12, fontweight='bold')
        plt.xticks(rotation=30, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on top of bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, "all_exercises_accuracy_comparison.png"), dpi=300, bbox_inches='tight')
        plt.show()
    # ========== END COMPARISON CHART ==========
    
    return results


if __name__ == "__main__":
    print("="*50)
    print("POSTURE CORRECTION MODEL TRAINING")
    print("="*50)
    
    print("\n[OPTION 1] Training combined model for all exercises...")
    X, y, exercise_labels = load_augmented_data()
    
    if X is not None:
        train_model(X, y, exercise_labels)
    
    print("\n[OPTION 2] Training separate models for each exercise...")
    results = train_exercise_specific_models()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    
    print("\nExercise-specific model accuracies:")
    for exercise, acc in results.items():
        print(f"  - {exercise}: {acc*100:.2f}%")