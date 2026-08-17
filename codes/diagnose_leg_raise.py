import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib


def angle_3pt(a, b, c):
    """Stable angle at joint b using dot product."""
    va = np.array([a.x - b.x, a.y - b.y])
    vc = np.array([c.x - b.x, c.y - b.y])
    cos_a = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def diagnose_leg_raise():
    model  = joblib.load("models/leg_raise_model.pkl")
    scaler = joblib.load("models/leg_raise_scaler.pkl")

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

    print("[INFO] Starting LEG RAISE diagnosis ...")
    print("[INFO] Press 'q' to quit\n")
    print("WHAT TO WATCH:")
    print("  • Knee angle    → should be >155° (straight leg)")
    print("  • Hip angle     → should decrease as leg rises")
    print("  • Ankle < Hip Y → ankle must be ABOVE hip in image coords")
    print("  • Visibility    → all key joints must be > 0.5\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result   = landmarker.detect(mp_image)

        # ── dark panel background ──────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (420, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (420, 0), (420, h), (200, 130, 60), 1)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            # ── draw skeleton dots ─────────────────────────────────────────
            for p in lm:
                cx, cy = int(p.x * w), int(p.y * h)
                cv2.circle(frame, (cx, cy), 4, (80, 220, 100), -1)

            # ── key landmarks ──────────────────────────────────────────────
            # Left side  (indices 23, 25, 27)
            l_hip, l_knee, l_ankle = lm[23], lm[25], lm[27]
            # Right side (indices 24, 26, 28)
            r_hip, r_knee, r_ankle = lm[24], lm[26], lm[28]

            # ── angles ────────────────────────────────────────────────────
            l_knee_ang = angle_3pt(l_hip, l_knee, l_ankle)
            r_knee_ang = angle_3pt(r_hip, r_knee, r_ankle)

            # Hip flexion: angle between torso (shoulder→hip) and thigh (hip→knee)
            l_shoulder = lm[11]
            r_shoulder = lm[12]
            l_hip_ang  = angle_3pt(l_shoulder, l_hip, l_knee)
            r_hip_ang  = angle_3pt(r_shoulder, r_hip, r_knee)

            # ── rule checks (LEFT leg — change to right if needed) ─────────
            l_raised      = l_ankle.y < l_hip.y - 0.05   # ankle above hip
            l_straight    = l_knee_ang > 155
            l_vis_ok      = min(l_hip.visibility, l_knee.visibility, l_ankle.visibility) > 0.5

            r_raised      = r_ankle.y < r_hip.y - 0.05
            r_straight    = r_knee_ang > 155
            r_vis_ok      = min(r_hip.visibility, r_knee.visibility, r_ankle.visibility) > 0.5

            # ── ML prediction ─────────────────────────────────────────────
            kp     = np.array([[p.x, p.y, p.z] for p in lm]).flatten().reshape(1, -1)
            kp_sc  = scaler.transform(kp)
            pred   = model.predict(kp_sc)[0]
            probs  = model.predict_proba(kp_sc)[0]

            # ── draw skeleton lines for legs ───────────────────────────────
            LEGS = [(23,25),(25,27),(24,26),(26,28),(23,24)]
            for a, b in LEGS:
                pa = (int(lm[a].x*w), int(lm[a].y*h))
                pb = (int(lm[b].x*w), int(lm[b].y*h))
                cv2.line(frame, pa, pb, (200, 130, 60), 2, cv2.LINE_AA)

            # ── HUD text ───────────────────────────────────────────────────
            def row(txt, y, col=(240,240,240)):
                cv2.putText(frame, txt, (12, y), FONT, 0.58, col, 1, cv2.LINE_AA)

            pred_col = (80,220,100) if pred == 1 else (60,70,230)
            cv2.putText(frame, f"ML: {'CORRECT' if pred==1 else 'INCORRECT'}",
                        (12, 40), FONT, 1.0, pred_col, 2, cv2.LINE_AA)

            row(f"Incorrect prob : {probs[0]*100:.1f}%", 75,  (60,70,230))
            row(f"Correct   prob : {probs[1]*100:.1f}%", 100, (80,220,100))

            cv2.line(frame, (12, 115), (400, 115), (60,60,70), 1)

            # Left leg
            cv2.putText(frame, "LEFT LEG", (12, 140), FONT, 0.65, (30,200,240), 2)
            lc = lambda ok: (80,220,100) if ok else (60,70,230)
            row(f"Knee angle  : {l_knee_ang:.1f}°  (need >155)", 165, lc(l_straight))
            row(f"Hip  angle  : {l_hip_ang:.1f}°",               190)
            row(f"Ankle < Hip : {l_raised}  (ankle.y={l_ankle.y:.3f}  hip.y={l_hip.y:.3f})", 215, lc(l_raised))
            row(f"Visibility  : {l_vis_ok}  (min={min(l_hip.visibility,l_knee.visibility,l_ankle.visibility):.2f})", 240, lc(l_vis_ok))
            row(f"RULE PASS   : {l_raised and l_straight and l_vis_ok}", 265, lc(l_raised and l_straight and l_vis_ok))

            cv2.line(frame, (12, 280), (400, 280), (60,60,70), 1)

            # Right leg
            cv2.putText(frame, "RIGHT LEG", (12, 305), FONT, 0.65, (30,200,240), 2)
            row(f"Knee angle  : {r_knee_ang:.1f}°  (need >155)", 330, lc(r_straight))
            row(f"Hip  angle  : {r_hip_ang:.1f}°",               355)
            row(f"Ankle < Hip : {r_raised}  (ankle.y={r_ankle.y:.3f}  hip.y={r_hip.y:.3f})", 380, lc(r_raised))
            row(f"Visibility  : {r_vis_ok}  (min={min(r_hip.visibility,r_knee.visibility,r_ankle.visibility):.2f})", 405, lc(r_vis_ok))
            row(f"RULE PASS   : {r_raised and r_straight and r_vis_ok}", 430, lc(r_raised and r_straight and r_vis_ok))

            cv2.line(frame, (12, 445), (400, 445), (60,60,70), 1)

            # ── Verdict ───────────────────────────────────────────────────
            either_rule = (l_raised and l_straight) or (r_raised and r_straight)
            verdict     = pred == 1 and either_rule
            v_col       = (80,220,100) if verdict else (60,70,230)
            cv2.putText(frame, f"COMBINED: {'CORRECT' if verdict else 'INCORRECT'}",
                        (12, 480), FONT, 0.85, v_col, 2, cv2.LINE_AA)

            # ── What's failing ────────────────────────────────────────────
            issues = []
            if probs[1] < 0.75:  issues.append(f"Low ML conf ({probs[1]*100:.0f}%)")
            if not l_straight and not r_straight:
                issues.append(f"Leg not straight ({max(l_knee_ang,r_knee_ang):.0f}°)")
            if not l_raised and not r_raised:
                issues.append("Leg not raised high enough")
            if not l_vis_ok and not r_vis_ok:
                issues.append("Landmarks not visible")

            y_off = 515
            cv2.putText(frame, "ISSUES:" if issues else "No issues found",
                        (12, y_off), FONT, 0.55, (240,200,60) if issues else (80,220,100), 1)
            for iss in issues:
                y_off += 22
                cv2.putText(frame, f"  • {iss}", (12, y_off), FONT, 0.5, (240,200,60), 1)

        else:
            cv2.putText(frame, "No pose detected — step back",
                        (12, 50), FONT, 0.8, (60,70,230), 2, cv2.LINE_AA)

        cv2.putText(frame, "LEG RAISE DIAGNOSIS  |  Q to quit",
                    (12, h - 12), FONT, 0.45, (130,130,145), 1)

        cv2.imshow("Leg Raise Diagnosis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    diagnose_leg_raise()