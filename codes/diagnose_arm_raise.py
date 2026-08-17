import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib

def diagnose_arm_raise():
    """Diagnose arm raise model predictions using NEW MediaPipe API."""

    # Load model
    model = joblib.load("models/arm_raise_model.pkl")
    scaler = joblib.load("models/arm_raise_scaler.pkl")

    # ✅ NEW MediaPipe PoseLandmarker
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    print("[INFO] Starting ARM RAISE diagnosis...")
    print("[INFO] Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Detect pose
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # Extract keypoints
            keypoints = []
            for lm in landmarks:
                keypoints.extend([lm.x, lm.y, lm.z])
            keypoints = np.array(keypoints).reshape(1, -1)

            # Predict
            keypoints_scaled = scaler.transform(keypoints)
            prediction = model.predict(keypoints_scaled)[0]
            probabilities = model.predict_proba(keypoints_scaled)[0]

            # Draw simple circles (since drawing_utils not available here)
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            # Show prediction
            cv2.putText(frame, f"Prediction: {'CORRECT' if prediction == 1 else 'INCORRECT'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Incorrect: {probabilities[0]*100:.1f}%",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"Correct: {probabilities[1]*100:.1f}%",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Manual check (arm raise logic)
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            left_raised = left_wrist.y < left_shoulder.y
            right_raised = right_wrist.y < right_shoulder.y

            cv2.putText(frame, f"Left arm raised: {left_raised}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"Right arm raised: {right_raised}",
                        (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        else:
            cv2.putText(frame, "No pose detected",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("ARM RAISE Diagnosis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    diagnose_arm_raise()