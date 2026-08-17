import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import os
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from voice_feedback import speak


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.55
SMOOTH_WINDOW   = 12
REP_HOLD_FRAMES = 6
FONT            = cv2.FONT_HERSHEY_DUPLEX

C_PANEL  = (12, 10, 18)
C_GREEN  = (80, 220, 120)
C_YELLOW = (60, 200, 240)
C_CYAN   = (220, 180, 80)
C_WHITE  = (240, 240, 240)
C_GRAY   = (140, 140, 150)
C_ORANGE = (60, 140, 255)
C_ACCENT = (100, 220, 180)


def angle_3pt(a, b, c):
    va = np.array([a.x - b.x, a.y - b.y])
    vc = np.array([c.x - b.x, c.y - b.y])
    cos_a = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


# ─────────────────────────────────────────────────────────────────────────────
# SMART DELTA-BASED FEEDBACK
# Driven by how much the person has MOVED from their own baseline,
# not by fixed angle thresholds. This means it works for every body type.
# ─────────────────────────────────────────────────────────────────────────────
def get_smart_feedback(exercise, angle, direction, baseline_angle, peak_angle, rep_count):
    if baseline_angle is None:
        return "Getting ready..."

    if exercise in ["arm_raise", "lateral_raise"]:
        travel = angle - baseline_angle           # positive = arm going up
        denom  = max(peak_angle - baseline_angle, 30)

        if direction == "up":
            if travel < 15:
                return "Start raising your arm"
            elif travel < 35:
                return "Keep lifting"
            elif travel < 55:
                return "Looking good, keep going"
            elif travel < 75:
                return "Almost at the top"
            else:
                return "Great — hold it there"
        elif direction == "down":
            remaining = angle - baseline_angle
            if remaining > 50:
                return "Lower slowly"
            elif remaining > 20:
                return "Controlled descent"
            else:
                return f"Good rep! {rep_count} done" if rep_count > 0 else "Reset — ready"
        else:
            if travel > 55:
                return "Perfect — hold this position"
            elif travel < 10:
                return "Raise your arm when ready"
            else:
                return "Keep lifting"

    elif exercise == "squat":
        travel = baseline_angle - angle           # positive = going deeper

        if direction == "down":
            if travel < 10:
                return "Start bending your knees"
            elif travel < 25:
                return "Go lower"
            elif travel < 45:
                return "Good depth, keep going"
            elif travel < 65:
                return "Almost there"
            else:
                return "Perfect depth — hold"
        elif direction == "up":
            remaining = baseline_angle - angle
            if remaining > 50:
                return "Drive up through your heels"
            elif remaining > 20:
                return "Nearly standing"
            else:
                return f"Strong rep! {rep_count} done" if rep_count > 0 else "Stand tall"
        else:
            if travel > 45:
                return "Hold the squat"
            elif travel < 10:
                return "Bend your knees to begin"
            else:
                return "Keep your back straight"

    elif exercise == "leg_raise":
        travel = baseline_angle - angle           # positive = leg rising

        if direction == "up":
            if travel < 10:
                return "Raise your knee"
            elif travel < 25:
                return "Lift higher"
            elif travel < 45:
                return "Good height, keep going"
            else:
                return "Hold — feel the contraction"
        elif direction == "down":
            remaining = baseline_angle - angle
            if remaining > 30:
                return "Lower with control"
            elif remaining > 10:
                return "Slow and steady"
            else:
                return f"Nice rep! {rep_count} done" if rep_count > 0 else "Reset — ready"
        else:
            if travel > 35:
                return "Hold — feel the contraction"
            elif travel < 10:
                return "Lift your knee when ready"
            else:
                return "Maintain the position"

    return "Move slowly and deliberately"


# ─────────────────────────────────────────────────────────────────────────────
# GUIDE SILHOUETTE
# ─────────────────────────────────────────────────────────────────────────────
def draw_guide(frame, exercise):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cx = w // 2
    ghost_col  = (100, 100, 120)
    target_col = (80, 200, 130)
    tk = 2

    head_y = int(h * 0.17)
    head_r = int(h * 0.042)
    ls = (int(cx - w * 0.09), int(h * 0.27))
    rs = (int(cx + w * 0.09), int(h * 0.27))
    lh = (int(cx - w * 0.065), int(h * 0.50))
    rh = (int(cx + w * 0.065), int(h * 0.50))
    lk = (int(cx - w * 0.075), int(h * 0.69))
    rk = (int(cx + w * 0.075), int(h * 0.69))
    lf = (int(cx - w * 0.085), int(h * 0.87))
    rf = (int(cx + w * 0.085), int(h * 0.87))

    cv2.circle(overlay, (cx, head_y), head_r, ghost_col, 1)
    for a, b in [(ls, rs), (ls, lh), (rs, rh), (lh, rh),
                 (lh, lk), (rh, rk), (lk, lf), (rk, rf)]:
        cv2.line(overlay, a, b, ghost_col, tk, cv2.LINE_AA)

    if exercise in ["lateral_raise", "arm_raise"]:
        ty = ls[1]
        le = (int(cx - w * 0.26), ty)
        re = (int(cx + w * 0.26), ty)
        cv2.line(overlay, ls, le, target_col, 3, cv2.LINE_AA)
        cv2.line(overlay, rs, re, target_col, 3, cv2.LINE_AA)
        cv2.circle(overlay, le, 7, target_col, -1)
        cv2.circle(overlay, re, 7, target_col, -1)
        for x in range(le[0], re[0], 16):
            cv2.line(overlay, (x, ty - 18), (min(x + 9, re[0]), ty - 18),
                     target_col, 1, cv2.LINE_AA)
        cv2.putText(overlay, "TARGET: shoulder height",
                    (int(w * 0.34), ty - 25), FONT, 0.38, target_col, 1, cv2.LINE_AA)

    elif exercise == "squat":
        lh_t = (int(cx - w * 0.09), int(h * 0.45))
        rh_t = (int(cx + w * 0.09), int(h * 0.45))
        lk_t = (int(cx - w * 0.10), int(h * 0.60))
        rk_t = (int(cx + w * 0.10), int(h * 0.60))
        lf_t = (int(cx - w * 0.12), int(h * 0.82))
        rf_t = (int(cx + w * 0.12), int(h * 0.82))
        for a, b in [(lh_t, lk_t), (rh_t, rk_t), (lk_t, lf_t),
                     (rk_t, rf_t), (lh_t, rh_t)]:
            cv2.line(overlay, a, b, target_col, 3, cv2.LINE_AA)
        cv2.circle(overlay, lk_t, 7, target_col, -1)
        cv2.circle(overlay, rk_t, 7, target_col, -1)
        cv2.putText(overlay, "TARGET: 90deg bend",
                    (int(w * 0.36), int(h * 0.88)), FONT, 0.38, target_col, 1, cv2.LINE_AA)

    elif exercise == "leg_raise":
        lh_p = (int(cx - w * 0.065), int(h * 0.50))
        lk_r = (int(cx - w * 0.03),  int(h * 0.40))
        cv2.line(overlay, lh_p, lk_r, target_col, 3, cv2.LINE_AA)
        cv2.circle(overlay, lk_r, 7, target_col, -1)
        cv2.arrowedLine(overlay,
                        (int(cx - w * 0.06), int(h * 0.62)),
                        (int(cx - w * 0.04), int(h * 0.42)),
                        target_col, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(overlay, "TARGET: hip height",
                    (int(w * 0.36), int(h * 0.38)), FONT, 0.38, target_col, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.38, frame, 0.62, 0, frame)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# ARC PROGRESS RING
# ─────────────────────────────────────────────────────────────────────────────
def draw_arc_ring(frame, cx, cy, radius, progress, color, bg_color=(50, 50, 60)):
    cv2.circle(frame, (cx, cy), radius, bg_color, 3)
    if progress > 0:
        end_angle = int(-90 + 360 * min(progress, 1.0))
        cv2.ellipse(frame, (cx, cy), (radius, radius),
                    0, -90, end_angle, color, 3, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def rule_check(exercise, lm):
    if exercise == "arm_raise":
        l_ang    = angle_3pt(lm[11], lm[13], lm[15])
        r_ang    = angle_3pt(lm[12], lm[14], lm[16])
        l_raised = lm[15].y < lm[11].y - 0.02
        r_raised = lm[16].y < lm[12].y - 0.02
        angle    = l_ang if (l_raised and not r_raised) else (r_ang if r_raised else max(l_ang, r_ang))
        if (l_raised and l_ang > 150) or (r_raised and r_ang > 150):
            return True, "", angle
        issues = []
        if max(l_ang, r_ang) <= 150:
            issues.append(f"Elbow bent ({max(l_ang, r_ang):.0f})")
        if not l_raised and not r_raised:
            issues.append("Wrist not above shoulder")
        return False, " | ".join(issues), angle

    if exercise == "leg_raise":
        l_hip = angle_3pt(lm[11], lm[23], lm[25])
        r_hip = angle_3pt(lm[12], lm[24], lm[26])
        angle = min(l_hip, r_hip)
        if l_hip < 130 or r_hip < 130:
            return True, "", angle
        return False, "Raise left higher" if l_hip < r_hip else "Raise right higher", angle

    if exercise == "squat":
        l_ang = angle_3pt(lm[23], lm[25], lm[27])
        r_ang = angle_3pt(lm[24], lm[26], lm[28])
        angle = min(l_ang, r_ang)
        if 70 <= angle <= 135:
            return True, "", angle
        return False, f"Bend more ({angle:.0f})", angle

    if exercise == "lateral_raise":
        l_ang    = angle_3pt(lm[11], lm[13], lm[15])
        r_ang    = angle_3pt(lm[12], lm[14], lm[16])
        angle    = max(l_ang, r_ang)
        l_raised = lm[13].y < lm[11].y + 0.05
        r_raised = lm[14].y < lm[12].y + 0.05
        if l_raised or r_raised:
            return True, "", angle
        return False, "Lift to shoulder height", angle

    return True, "", 90


def visibility_ok(exercise, lm):
    if exercise == "arm_raise":
        return (all(lm[i].visibility > 0.5 for i in [11, 13, 15]) or
                all(lm[i].visibility > 0.5 for i in [12, 14, 16]))
    if exercise == "leg_raise":
        return (all(lm[i].visibility > 0.4 for i in [23, 25]) or
                all(lm[i].visibility > 0.4 for i in [24, 26]))
    if exercise == "lateral_raise":
        return (all(lm[i].visibility > 0.5 for i in [11, 13]) or
                all(lm[i].visibility > 0.5 for i in [12, 14]))
    return all(lm[i].visibility > 0.4 for i in [23, 24, 25, 26, 27, 28])


def get_display_angle(exercise, lm, w, h):
    if exercise == "arm_raise":
        l = angle_3pt(lm[11], lm[13], lm[15])
        r = angle_3pt(lm[12], lm[14], lm[16])
        return (l, lm_to_px(lm[13], w, h)) if lm[15].y < lm[16].y else (r, lm_to_px(lm[14], w, h))
    if exercise == "leg_raise":
        l = angle_3pt(lm[11], lm[23], lm[25])
        r = angle_3pt(lm[12], lm[24], lm[26])
        return (l, lm_to_px(lm[25], w, h)) if l < r else (r, lm_to_px(lm[26], w, h))
    if exercise == "lateral_raise":
        return (angle_3pt(lm[11], lm[13], lm[15]), lm_to_px(lm[13], w, h)) if lm[13].y < lm[14].y \
            else (angle_3pt(lm[12], lm[14], lm[16]), lm_to_px(lm[14], w, h))
    l = angle_3pt(lm[23], lm[25], lm[27])
    r = angle_3pt(lm[24], lm[26], lm[28])
    return (l, lm_to_px(lm[25], w, h)) if l < r else (r, lm_to_px(lm[26], w, h))


# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM HUD
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(frame, exercise, correct, conf, rep_count,
             acc, fps, hold_counter, angle, direction,
             baseline_angle, peak_angle):
    h, w = frame.shape[:2]

    # Top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 105), C_PANEL, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    accent_line_col = C_GREEN if correct else (80, 80, 200)
    cv2.line(frame, (0, 105), (w, 105), accent_line_col, 2, cv2.LINE_AA)

    # Exercise name
    cv2.putText(frame, exercise.replace("_", " ").upper(),
                (18, 38), FONT, 0.80, C_CYAN, 1, cv2.LINE_AA)

    # Direction pill
    dir_map = {"up": "RISING", "down": "LOWERING", None: "HOLD"}
    dir_col = (80, 220, 120) if direction == "up" else \
              ((80, 130, 220) if direction == "down" else C_GRAY)
    cv2.putText(frame, dir_map.get(direction, "HOLD"),
                (220, 38), FONT, 0.46, dir_col, 1, cv2.LINE_AA)

    # Form status
    cv2.putText(frame, "FORM OK" if correct else "ADJUST FORM",
                (18, 76), FONT, 0.92, C_GREEN if correct else C_YELLOW, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(angle)} deg",
                (18, 100), FONT, 0.44, C_WHITE, 1, cv2.LINE_AA)

    # Right: reps + accuracy
    rx = w - 170
    cv2.putText(frame, str(rep_count), (rx, 55), FONT, 1.55, C_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "REPS", (rx + 50, 55), FONT, 0.38, C_GRAY, 1, cv2.LINE_AA)
    acc_col = C_GREEN if acc > 75 else (C_YELLOW if acc > 50 else (80, 80, 220))
    cv2.putText(frame, f"{acc:.0f}% ACC", (rx, 78), FONT, 0.40, acc_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f} FPS", (rx, 98), FONT, 0.36, C_GRAY, 1, cv2.LINE_AA)

    # Rep ring (bottom left)
    ring_x, ring_y, ring_r = 38, h - 50, 26
    prog_h = min(hold_counter / REP_HOLD_FRAMES, 1.0)
    draw_arc_ring(frame, ring_x, ring_y, ring_r, prog_h,
                  C_GREEN if prog_h >= 1.0 else C_ACCENT)
    label = "REP!" if prog_h >= 1.0 else f"{int(prog_h * 100)}%"
    cv2.putText(frame, label,
                (ring_x - 18 if prog_h >= 1.0 else ring_x - 12, ring_y + 5),
                FONT, 0.34, C_WHITE, 1, cv2.LINE_AA)

    # Movement progress bar (bottom full-width)
    bar_y, bar_h, pad = h - 12, 6, 80
    if baseline_angle is not None and peak_angle is not None:
        if exercise in ["arm_raise", "lateral_raise"]:
            travel = max(angle - baseline_angle, 0)
            denom  = max(peak_angle - baseline_angle, 30)
        else:
            travel = max(baseline_angle - angle, 0)
            denom  = max(baseline_angle - peak_angle, 30)
        mov_prog = min(travel / denom, 1.0)
    else:
        mov_prog = 0

    bw = w - pad * 2
    cv2.rectangle(frame, (pad, bar_y), (pad + bw, bar_y + bar_h), (40, 40, 50), -1)
    if mov_prog > 0:
        cv2.rectangle(frame, (pad, bar_y),
                      (pad + int(bw * mov_prog), bar_y + bar_h),
                      C_GREEN if correct else C_YELLOW, -1)
    cv2.putText(frame, "REST", (8, bar_y + 5), FONT, 0.30, C_GRAY, 1)
    cv2.putText(frame, "PEAK", (w - 46, bar_y + 5), FONT, 0.30, C_GRAY, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────
class RehabSystem:

    def __init__(self, exercise: str):
        self.exercise = exercise

        self.model = joblib.load(f"models/{exercise}_model.pkl")
        sp = f"models/{exercise}_scaler.pkl"
        if not os.path.exists(sp):
            sp = "models/scaler.pkl"
        self.scaler = joblib.load(sp)

        base = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.pose = vision.PoseLandmarker.create_from_options(opts)

        self.pred_history   = deque(maxlen=SMOOTH_WINDOW)
        self.pred_buffer    = deque(maxlen=10)
        self.hold_counter   = 0
        self.rep_count      = 0
        self.in_rep         = False
        self.session_start  = time.time()
        self.total_frames   = 0
        self.correct_frames = 0
        self.fps_history    = deque(maxlen=30)
        self._last_time     = time.time()
        self._last_correct  = False
        self.current_feedback = ""

        self.prev_angle = None
        self.direction  = None
        self.phase      = "start"

        # Calibration
        self._calib_angles  = []
        self._calib_done    = False
        self.baseline_angle = None
        self.peak_angle     = None

    def get_pose(self, frame):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        ts  = int((time.time() - self.session_start) * 1000)
        res = self.pose.detect_for_video(img, ts)
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            kp = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
            return kp, lm
        return None, None

    def smooth_pred(self, pred):
        if self.exercise == "lateral_raise":
            self.pred_buffer.append(pred)
            return int(sum(self.pred_buffer) > len(self.pred_buffer) * 0.6)
        self.pred_history.append(pred)
        ones = self.pred_history.count(1)
        return 1 if ones > len(self.pred_history) - ones else 0

    def update_reps(self, correct):
        if correct:
            self.hold_counter += 1
            if self.hold_counter >= REP_HOLD_FRAMES and not self.in_rep:
                self.rep_count += 1
                self.in_rep = True
        else:
            self.hold_counter = 0
            self.in_rep = False

    def update_movement(self, angle):
        DEAD_ZONE = 1.8

        # Calibration: first 40 frames at rest
        if not self._calib_done:
            self._calib_angles.append(angle)
            if len(self._calib_angles) >= 40:
                self.baseline_angle = float(np.median(self._calib_angles))
                self.peak_angle     = self.baseline_angle
                self._calib_done    = True
            self.prev_angle = angle
            return

        if self.prev_angle is not None:
            delta = angle - self.prev_angle
            if delta > DEAD_ZONE:
                self.direction = "up"
            elif delta < -DEAD_ZONE:
                self.direction = "down"

        self.prev_angle = angle

        # Track peak and reset when back at rest
        if self.exercise in ["arm_raise", "lateral_raise"]:
            if self.direction == "up":
                self.peak_angle = max(self.peak_angle, angle)
            if angle <= self.baseline_angle + 8:
                self.peak_angle = self.baseline_angle
        elif self.exercise == "squat":
            if self.direction == "down":
                self.peak_angle = min(self.peak_angle, angle)
            if angle >= self.baseline_angle - 8:
                self.peak_angle = self.baseline_angle
        elif self.exercise == "leg_raise":
            if self.direction == "up":
                self.peak_angle = min(self.peak_angle, angle)
            if angle >= self.baseline_angle - 8:
                self.peak_angle = self.baseline_angle

    def draw_skeleton(self, frame, lm):
        h, w = frame.shape[:2]
        connections = [
            (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 12), (11, 23), (12, 24),
            (23, 25), (25, 27), (24, 26), (26, 28), (23, 24),
        ]
        color = C_GREEN if self._last_correct else (100, 100, 220)
        for s, e in connections:
            if s < len(lm) and e < len(lm):
                cv2.line(frame,
                         (int(lm[s].x * w), int(lm[s].y * h)),
                         (int(lm[e].x * w), int(lm[e].y * h)),
                         color, 3, cv2.LINE_AA)
        for i in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if i < len(lm):
                px = (int(lm[i].x * w), int(lm[i].y * h))
                cv2.circle(frame, px, 6, color, -1)
                cv2.circle(frame, px, 7, C_WHITE, 1)

    def draw_calibration(self, frame, n):
        h, w = frame.shape[:2]
        frame[:] = (12, 10, 18)
        cx, cy = w // 2, h // 2
        draw_arc_ring(frame, cx, cy, 60, n / 40, C_ACCENT, (40, 40, 50))
        cv2.putText(frame, f"{int(n / 40 * 100)}%",
                    (cx - 22, cy + 10), FONT, 0.9, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, "CALIBRATING — stand naturally",
                    (int(w * 0.22), cy + 90), FONT, 0.50, C_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, "Your resting position is being recorded",
                    (int(w * 0.27), cy + 118), FONT, 0.40, (100, 100, 120), 1, cv2.LINE_AA)

    def process_frame_with_feedback(self, frame):
        processed = self.process_frame(frame)
        accuracy  = (self.correct_frames / self.total_frames * 100) if self.total_frames else 0
        return processed, self.current_feedback, accuracy

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        now = time.time()
        self.fps_history.append(1.0 / max(now - self._last_time, 1e-6))
        self._last_time = now
        fps = float(np.mean(self.fps_history))

        kp, lm = self.get_pose(frame)
        angle  = self.prev_angle if self.prev_angle else 90

        if kp is not None and visibility_ok(self.exercise, lm):

            _, _, angle = rule_check(self.exercise, lm)

            # Calibration phase
            if not self._calib_done:
                self.update_movement(angle)
                self.draw_calibration(frame, len(self._calib_angles))
                self.current_feedback = "Stand naturally while we calibrate"
                return frame

            n     = self.scaler.n_features_in_
            kp_in = kp[:n] if kp.shape[0] >= n else np.pad(kp, (0, n - kp.shape[0]))
            X     = self.scaler.transform(kp_in.reshape(1, -1))
            pred  = self.model.predict(X)[0]
            prob  = self.model.predict_proba(X)[0]
            conf  = float(prob[pred])

            rule_ok, rule_reason, angle = rule_check(self.exercise, lm)

            if self.exercise in ["lateral_raise", "squat"]:
                correct = rule_ok
            else:
                correct = (pred == 1 and rule_ok) if conf >= CONF_THRESHOLD else False

            self._last_correct   = correct
            self.total_frames   += 1
            if correct:
                self.correct_frames += 1
            self.update_reps(correct)
            self.update_movement(angle)

            # Draw order: guide → skeleton → bubble → HUD
            draw_guide(frame, self.exercise)
            self.draw_skeleton(frame, lm)

            ang_val, ang_px = get_display_angle(self.exercise, lm, w, h)
            cv2.circle(frame, ang_px, 18, C_ORANGE, 2)
            cv2.putText(frame, f"{int(ang_val)}",
                        (ang_px[0] + 20, ang_px[1] - 4), FONT, 0.50, C_ORANGE, 1)

            acc = (self.correct_frames / self.total_frames * 100) if self.total_frames else 0
            draw_hud(frame, self.exercise, correct, conf,
                     self.rep_count, acc, fps, self.hold_counter,
                     angle, self.direction,
                     self.baseline_angle, self.peak_angle)

            feedback = get_smart_feedback(
                self.exercise, angle, self.direction,
                self.baseline_angle, self.peak_angle, self.rep_count
            )
            self.current_feedback = feedback
            speak(feedback)

        else:
            self.current_feedback = "Step back — keep full body visible"
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 105), C_PANEL, -1)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
            cv2.line(frame, (0, 105), (w, 105), (60, 60, 80), 2, cv2.LINE_AA)
            cv2.putText(frame, self.exercise.replace("_", " ").upper(),
                        (18, 38), FONT, 0.80, C_CYAN, 1, cv2.LINE_AA)
            cv2.putText(frame, "Step back — full body visible",
                        (18, 78), FONT, 0.78, (80, 165, 255), 1, cv2.LINE_AA)

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"\n[Rehab AI v6]  {self.exercise.upper()} — Press Q to quit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.process_frame(frame)
            cv2.imshow("Rehab AI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - self.session_start
        acc = (self.correct_frames / self.total_frames * 100) if self.total_frames else 0
        print(f"\n  Reps: {self.rep_count}  |  Accuracy: {acc:.1f}%  |"
              f"  Duration: {int(elapsed//60):02d}:{int(elapsed%60):02d}\n")


if __name__ == "__main__":
    mapping = {"1": "arm_raise", "2": "leg_raise", "3": "squat", "4": "lateral_raise"}
    print("\nREHAB AI v6\n1. Arm Raise  2. Leg Raise  3. Squat  4. Lateral Raise")
    choice = input("Choose [1-4]: ").strip()
    if choice not in mapping:
        exit("Invalid choice")
    RehabSystem(mapping[choice]).run()