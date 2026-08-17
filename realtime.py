import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("models/squat_model.pkl")
scaler = joblib.load("models/squat_scaler.pkl")

# -------------------------------
# MEDIAPIPE TASKS SETUP
# -------------------------------
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)

landmarker = vision.PoseLandmarker.create_from_options(options)

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(landmarks):
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return np.array(features)

# -------------------------------
# START CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not working")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Detect pose
    result = landmarker.detect(mp_image)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        features = extract_features(landmarks)

        if len(features) == 99:
            features = features.reshape(1, -1)
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]

            if prediction == 1:
                text = "Correct"
                color = (0, 255, 0)
            else:
                text = "Incorrect"
                color = (0, 0, 255)

            cv2.putText(frame, text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Posture Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()