import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib


def angle_3pt(a, b, c):
    va = np.array([a.x - b.x, a.y - b.y])
    vc = np.array([c.x - b.x, c.y - b.y])
    cos_a = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def diagnose_squat():
    model  = joblib.load("models/squat_model.pkl")
    scaler = joblib.load("models/squat_scaler.pkl")

    base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    print("[INFO] Starting SQUAT diagnosis ...")
    print("[INFO] Press 'q' to quit\n")
    print("WHAT TO WATCH:")
    print("  • Knee angle    → should be 65–115° at bottom of squat")
    print("  • Hip angle     → torso lean (shoulder→hip→knee)")
    print("  • Knee over toe → knee x should not go far past ankle x")
    print("  • Symmetry      → left vs right knee angle difference\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result   = landmarker.detect(mp_image)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (420, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (420, 0), (420, h), (200, 130, 60), 1)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            for p in lm:
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(frame, (cx, cy), 4, (80, 220, 100), -1)

            # Landmarks
            l_shoulder, r_shoulder = lm[11], lm[12]
            l_hip,  r_hip          = lm[23], lm[24]
            l_knee, r_knee         = lm[25], lm[26]
            l_ankle, r_ankle       = lm[27], lm[28]

            # Angles
            l_knee_ang = angle_3pt(l_hip,      l_knee,  l_ankle)
            r_knee_ang = angle_3pt(r_hip,       r_knee,  r_ankle)
            l_hip_ang  = angle_3pt(l_shoulder,  l_hip,   l_knee)
            r_hip_ang  = angle_3pt(r_shoulder,  r_hip,   r_knee)

            avg_knee   = (l_knee_ang + r_knee_ang) / 2
            symmetry   = abs(l_knee_ang - r_knee_ang)

            # Knee-over-toe (x axis: left side)
            l_knee_over = (l_knee.x - l_ankle.x)   # positive = forward
            r_knee_over = (r_knee.x - r_ankle.x)

            # Rule checks
            l_depth_ok   = 65 < l_knee_ang < 115
            r_depth_ok   = 65 < r_knee_ang < 115
            sym_ok       = symmetry < 20
            l_vis_ok     = min(l_hip.visibility, l_knee.visibility, l_ankle.visibility) > 0.5
            r_vis_ok     = min(r_hip.visibility, r_knee.visibility, r_ankle.visibility) > 0.5

            # ML prediction
            kp    = np.array([[p.x, p.y, p.z] for p in lm]).flatten().reshape(1, -1)
            kp_sc = scaler.transform(kp)
            pred  = model.predict(kp_sc)[0]
            probs = model.predict_proba(kp_sc)[0]

            # Skeleton legs
            LEGS = [(23,25),(25,27),(24,26),(26,28),(23,24),(11,23),(12,24),(11,12)]
            for a, b in LEGS:
                pa = (int(lm[a].x*w), int(lm[a].y*h))
                pb = (int(lm[b].x*w), int(lm[b].y*h))
                cv2.line(frame, pa, pb, (200,130,60), 2, cv2.LINE_AA)

            def row(txt, y, col=(240,240,240)):
                cv2.putText(frame, txt, (12, y), FONT, 0.58, col, 1, cv2.LINE_AA)

            lc = lambda ok: (80,220,100) if ok else (60,70,230)

            pred_col = (80,220,100) if pred == 1 else (60,70,230)
            cv2.putText(frame, f"ML: {'CORRECT' if pred==1 else 'INCORRECT'}",
                        (12, 40), FONT, 1.0, pred_col, 2, cv2.LINE_AA)
            row(f"Incorrect prob : {probs[0]*100:.1f}%", 75,  (60,70,230))
            row(f"Correct   prob : {probs[1]*100:.1f}%", 100, (80,220,100))

            cv2.line(frame, (12, 115), (400, 115), (60,60,70), 1)

            # Left leg
            cv2.putText(frame, "LEFT LEG", (12, 140), FONT, 0.65, (30,200,240), 2)
            row(f"Knee angle : {l_knee_ang:.1f}°  (need 65–115)",  165, lc(l_depth_ok))
            row(f"Hip  angle : {l_hip_ang:.1f}°",                  190)
            row(f"Knee fwd   : {l_knee_over:+.3f}  (neg=ok, +big=caving)", 215)
            row(f"Visibility : {l_vis_ok}  ({min(l_hip.visibility,l_knee.visibility,l_ankle.visibility):.2f})", 240, lc(l_vis_ok))

            cv2.line(frame, (12, 255), (400, 255), (60,60,70), 1)

            # Right leg
            cv2.putText(frame, "RIGHT LEG", (12, 280), FONT, 0.65, (30,200,240), 2)
            row(f"Knee angle : {r_knee_ang:.1f}°  (need 65–115)",  305, lc(r_depth_ok))
            row(f"Hip  angle : {r_hip_ang:.1f}°",                  330)
            row(f"Knee fwd   : {r_knee_over:+.3f}", 355)
            row(f"Visibility : {r_vis_ok}  ({min(r_hip.visibility,r_knee.visibility,r_ankle.visibility):.2f})", 380, lc(r_vis_ok))

            cv2.line(frame, (12, 395), (400, 395), (60,60,70), 1)

            # Summary
            cv2.putText(frame, "SUMMARY", (12, 420), FONT, 0.65, (30,200,240), 2)
            row(f"Avg knee angle : {avg_knee:.1f}°",               445)
            row(f"Symmetry diff  : {symmetry:.1f}°  (need <20)",   470, lc(sym_ok))

            either_rule = (l_depth_ok or r_depth_ok) and sym_ok
            verdict     = pred == 1 and either_rule
            v_col       = (80,220,100) if verdict else (60,70,230)
            cv2.putText(frame, f"COMBINED: {'CORRECT' if verdict else 'INCORRECT'}",
                        (12, 505), FONT, 0.85, v_col, 2, cv2.LINE_AA)

            # Issues
            issues = []
            if probs[1] < 0.75:  issues.append(f"Low ML conf ({probs[1]*100:.0f}%)")
            if not l_depth_ok and not r_depth_ok:
                issues.append(f"Not deep enough / too deep ({avg_knee:.0f}°, need 65–115)")
            if not sym_ok:
                issues.append(f"Asymmetric squat (L={l_knee_ang:.0f}° R={r_knee_ang:.0f}°)")
            if not l_vis_ok or not r_vis_ok:
                issues.append("Key landmarks not visible")
            if abs(l_knee_over) > 0.08 or abs(r_knee_over) > 0.08:
                issues.append("Knees caving in / too far forward")

            y_off = 535
            cv2.putText(frame, "ISSUES:" if issues else "No issues found",
                        (12, y_off), FONT, 0.55, (240,200,60) if issues else (80,220,100), 1)
            for iss in issues:
                y_off += 22
                cv2.putText(frame, f"  • {iss}", (12, y_off), FONT, 0.5, (240,200,60), 1)

        else:
            cv2.putText(frame, "No pose detected — step back",
                        (12, 50), FONT, 0.8, (60,70,230), 2, cv2.LINE_AA)

        cv2.putText(frame, "SQUAT DIAGNOSIS  |  Q to quit",
                    (12, h - 12), FONT, 0.45, (130,130,145), 1)

        cv2.imshow("Squat Diagnosis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    diagnose_squat()