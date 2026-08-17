import numpy as np
import streamlit as st
import cv2
import time
from rehabilitation_system import RehabSystem

st.set_page_config(
    page_title="Rehab AI Trainer",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM UI STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #080c14;
    color: #e2e8f0;
}

/* ── Header ──────────────────────────────────────────────────────────────── */
.rehab-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 24px 0 8px 0;
    border-bottom: 1px solid rgba(59,130,246,0.2);
    margin-bottom: 24px;
}
.rehab-logo {
    font-size: 40px;
}
.rehab-title {
    font-size: 30px;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin: 0;
}
.rehab-sub {
    font-size: 13px;
    color: #64748b;
    margin: 2px 0 0 0;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
div.stButton > button {
    width: 100%;
    min-height: 110px;
    border-radius: 14px;
    font-size: 16px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
    border: 1px solid rgba(59,130,246,0.25);
    transition: all 0.25s ease;
    white-space: pre-line;
    padding: 20px 16px;
    line-height: 1.5;
    letter-spacing: 0.1px;
}
div.stButton > button:hover {
    transform: translateY(-3px);
    border-color: rgba(96,165,250,0.6);
    box-shadow: 0 8px 24px rgba(59,130,246,0.2);
    background: linear-gradient(135deg, #1e293b, #263451);
    color: #ffffff;
}
div.stButton > button:active {
    transform: translateY(0px);
}

/* ── Selectbox ──────────────────────────────────────────────────────────── */
div.stSelectbox > div > div {
    background-color: #0f172a;
    color: #e2e8f0;
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 10px;
}

/* ── Text input ─────────────────────────────────────────────────────────── */
div.stTextInput > div > div > input {
    background-color: #0f172a;
    color: #e2e8f0;
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 10px;
    padding: 10px 14px;
    font-family: 'Space Grotesk', sans-serif;
}
div.stTextInput > div > div > input:focus {
    border-color: rgba(96,165,250,0.7);
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15);
}

/* ── Feedback cards ─────────────────────────────────────────────────────── */
.feedback-card {
    border-radius: 14px;
    padding: 18px 22px;
    margin: 12px 0;
    font-size: 20px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 12px;
    letter-spacing: -0.2px;
    animation: fadeSlide 0.35s ease;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
}
.feedback-success {
    background: linear-gradient(135deg, #052e16, #065f46);
    border: 1px solid #10b981;
    color: #6ee7b7;
}
.feedback-warning {
    background: linear-gradient(135deg, #451a03, #78350f);
    border: 1px solid #f59e0b;
    color: #fcd34d;
}
.feedback-info {
    background: linear-gradient(135deg, #0c1a2e, #1e3a5f);
    border: 1px solid #3b82f6;
    color: #93c5fd;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 12px 0;
}
.metric-card {
    flex: 1;
    background: linear-gradient(135deg, #0f172a, #1a2440);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 14px;
    padding: 16px 12px;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover {
    border-color: rgba(96,165,250,0.4);
}
.metric-value {
    font-size: 34px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
.metric-label {
    font-size: 12px;
    color: #475569;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* ── Phase badge ─────────────────────────────────────────────────────────── */
.phase-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.35);
    color: #93c5fd;
    margin-right: 8px;
}

/* ── Tip bar ────────────────────────────────────────────────────────────── */
.tip-bar {
    background: rgba(15,23,42,0.8);
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 13px;
    color: #64748b;
    margin: 8px 0;
}

/* ── Step indicator ─────────────────────────────────────────────────────── */
.step-indicator {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
}
.step-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #1e3a5f;
    border: 1px solid rgba(59,130,246,0.3);
    transition: all 0.3s;
}
.step-dot.active {
    background: #3b82f6;
    box-shadow: 0 0 8px rgba(59,130,246,0.5);
}
.step-dot.done {
    background: #10b981;
}

/* ── Progress bar ───────────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #a78bfa) !important;
    border-radius: 4px;
}
.stProgress > div > div {
    background: #1e293b !important;
    border-radius: 4px;
}

/* ── Session summary ─────────────────────────────────────────────────────── */
.summary-card {
    background: linear-gradient(135deg, #0f172a, #1a2440);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin-top: 16px;
}
.summary-title {
    font-size: 22px;
    font-weight: 700;
    color: #6ee7b7;
    margin-bottom: 8px;
}
.summary-sub {
    font-size: 14px;
    color: #475569;
}

/* ── Live indicator ─────────────────────────────────────────────────────── */
.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ef4444;
    margin-right: 6px;
    animation: pulse 1.2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
}

/* ── Divider ────────────────────────────────────────────────────────────── */
hr {
    border-color: rgba(59,130,246,0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rehab-header">
    <div class="rehab-logo">🏥</div>
    <div>
        <p class="rehab-title">Rehab AI Trainer</p>
        <p class="rehab-sub">Real-time posture correction · Voice guidance · Phase-aware feedback</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "Shoulder Pain":   ["arm_raise", "lateral_raise"],
    "Knee Pain":       ["leg_raise", "squat"],
    "General Fitness": ["arm_raise", "lateral_raise", "squat", "leg_raise"],
}

EXERCISE_META = {
    "arm_raise":     {"icon": "🙆", "desc": "Front raise for shoulder mobility"},
    "lateral_raise": {"icon": "🤸", "desc": "Side raise for deltoid strength"},
    "squat":         {"icon": "🏋️", "desc": "Knee bend for quad & glute strength"},
    "leg_raise":     {"icon": "🦵", "desc": "Hip flexion for core & hip strength"},
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "running": False,
    "system":  None,
    "step":    0,
    "user_name": "",
    "category": None,
    "selected_exercise": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# STEP DOTS HELPER
# ─────────────────────────────────────────────────────────────────────────────
def step_dots(current, total=4):
    dots = ""
    for i in range(total):
        cls = "active" if i == current else ("done" if i < current else "")
        dots += f'<div class="step-dot {cls}"></div>'
    st.markdown(f'<div class="step-indicator">{dots}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEEDBACK HELPER
# ─────────────────────────────────────────────────────────────────────────────
def render_feedback(text: str):
    """Render colour-coded, animated feedback card."""
    if not text:
        return
    good_words  = ["Good", "Perfect", "hold", "rep", "Great", "Excellent"]
    warn_words  = ["Raise", "Lower", "Lift", "Almost", "Adjust", "Bend", "higher",
                   "slowly", "control", "Controlled", "Go lower", "Start"]

    if any(w in text for w in good_words):
        css = "feedback-success"
        icon = "✅"
    elif any(w in text for w in warn_words):
        css = "feedback-warning"
        icon = "⚡"
    else:
        css = "feedback-info"
        icon = "💡"

    st.markdown(f"""
    <div class="feedback-card {css}">
        <span style="font-size:24px">{icon}</span>
        <span>{text}</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROBUST CAMERA INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def init_camera():
    """Try multiple camera indices and return working camera."""
    for cam_idx in [0, 1, 2]:
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                return cap
            cap.release()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CHATBOT / ONBOARDING FLOW
# ─────────────────────────────────────────────────────────────────────────────

# ── STEP 0 — Name ────────────────────────────────────────────────────────────
if st.session_state.step == 0:
    step_dots(0)
    st.markdown("### 👋 Hi! I'm your Rehab AI Assistant")
    st.markdown("I'll guide your exercises with real-time voice coaching and posture correction.")

    name = st.text_input("What's your name?", placeholder="Enter your name to get started…")

    st.markdown("""
    <div class="tip-bar">
        💡 Make sure your camera is connected and you have ~6–8 ft of space in front of you.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Continue →", use_container_width=True) and name.strip():
        st.session_state.user_name = name.strip()
        st.session_state.step = 1
        st.rerun()


# ── STEP 1 — Category ────────────────────────────────────────────────────────
elif st.session_state.step == 1:
    step_dots(1)
    st.markdown(f"### Nice to meet you, **{st.session_state.user_name}** 🎯")
    st.markdown("What are you training for today?")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🦾 Shoulder Pain\nImprove mobility & range", use_container_width=True):
            st.session_state.category = "Shoulder Pain"
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("🦵 Knee Pain\nStrength & recovery", use_container_width=True):
            st.session_state.category = "Knee Pain"
            st.session_state.step = 2
            st.rerun()
    with col3:
        if st.button("🔥 General Fitness\nFull body training", use_container_width=True):
            st.session_state.category = "General Fitness"
            st.session_state.step = 2
            st.rerun()


# ── STEP 2 — Exercise selection ───────────────────────────────────────────────
elif st.session_state.step == 2:
    step_dots(2)
    st.markdown(f"### Great choice! 🎯  `{st.session_state.category}`")
    st.markdown("Pick an exercise to begin:")

    exercises = CATEGORY_MAP[st.session_state.category]
    selected = st.selectbox(
        "Exercise",
        exercises,
        format_func=lambda x: f"{EXERCISE_META[x]['icon']}  {x.replace('_', ' ').title()}"
    )

    # Show exercise info card
    meta = EXERCISE_META[selected]
    st.markdown(f"""
    <div class="tip-bar">
        {meta['icon']} <strong>{selected.replace('_', ' ').title()}</strong> — {meta['desc']}
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("▶  Start Exercise", use_container_width=True):
            st.session_state.selected_exercise = selected
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()


# ── STEP 3 — Pre-session checklist ───────────────────────────────────────────
elif st.session_state.step == 3:
    step_dots(3)
    ex   = st.session_state.selected_exercise
    meta = EXERCISE_META[ex]

    st.markdown(f"### Ready for  {meta['icon']}  **{ex.replace('_', ' ').title()}**")

    st.markdown("""
    <div style="background: #0f172a; border: 1px solid rgba(59,130,246,0.2); border-radius: 14px; padding: 20px 24px; margin: 12px 0;">
        <p style="color:#60a5fa; font-weight:700; margin:0 0 12px 0;">📋 Before you start</p>
        <p style="color:#94a3b8; margin:4px 0;">📸  Stand 6–8 feet from the camera</p>
        <p style="color:#94a3b8; margin:4px 0;">💡  Face a well-lit area</p>
        <p style="color:#94a3b8; margin:4px 0;">🔊  Turn on your speakers for voice coaching</p>
        <p style="color:#94a3b8; margin:4px 0;">👕  Wear fitted clothing for better tracking</p>
        <p style="color:#94a3b8; margin:4px 0; padding-top:8px; border-top:1px solid rgba(59,130,246,0.1); margin-top:8px;">💪  Move slowly — the AI reads small angle changes for coaching</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🎥  Begin Camera Session", use_container_width=True):
            st.session_state.running = True
            st.session_state.system  = RehabSystem(exercise=ex)
            st.session_state.step    = 4
            st.rerun()
    with col2:
        if st.button("🔄  Start Over", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — LIVE SESSION (WITH FIXED CAMERA HANDLING)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.step == 4:

    # ── Placeholders ─────────────────────────────────────────────────────────
    live_header      = st.empty()
    metrics_ph       = st.empty()
    progress_ph      = st.empty()
    feedback_ph      = st.empty()
    tip_ph           = st.empty()
    frame_ph         = st.empty()
    stop_col, _      = st.columns([1, 3])

    # ── Stop button ──────────────────────────────────────────────────────────
    with stop_col:
        stop_pressed = st.button("⏹  Stop Session", use_container_width=True)

    if stop_pressed:
        st.session_state.running = False

    if st.session_state.running and st.session_state.system:
        # Try to initialize camera with fallback indices
        cap = init_camera()
        
        if cap is None:
            st.error("⚠️ No camera found. Please check your camera connection and refresh the page.")
            st.session_state.running = False
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            live_header.markdown("""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                <span class="live-dot"></span>
                <span style="color:#ef4444; font-weight:700; font-size:13px; letter-spacing:1px;">LIVE</span>
                <span style="color:#475569; font-size:13px;">Camera session active</span>
            </div>
            """, unsafe_allow_html=True)

            tip_ph.markdown("""
            <div class="tip-bar">💡 Move slowly and deliberately — the AI tracks every degree of movement</div>
            """, unsafe_allow_html=True)

            frame_lost_count = 0
            MAX_FRAME_LOST = 5

            while st.session_state.running:
                ret, frame = cap.read()
                
                # Handle frame loss with retry
                if not ret:
                    frame_lost_count += 1
                    if frame_lost_count >= MAX_FRAME_LOST:
                        st.error("⚠️ Camera feed lost permanently. Please refresh the page.")
                        break
                    # Try to reinitialize camera
                    cap.release()
                    cap = init_camera()
                    if cap is None:
                        st.error("⚠️ Cannot reconnect to camera.")
                        break
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame_lost_count = 0

                processed, feedback_text, accuracy = st.session_state.system.process_frame_with_feedback(frame)
                system   = st.session_state.system
                acc      = (system.correct_frames / system.total_frames * 100) if system.total_frames > 0 else 0
                fps      = float(np.mean(system.fps_history)) if system.fps_history else 0
                duration = time.time() - system.session_start

                # ── Metrics row ──────────────────────────────────────────────────
                with metrics_ph.container():
                    st.markdown(f"""
                    <div class="metric-row">
                        <div class="metric-card">
                            <div class="metric-value">{system.rep_count}</div>
                            <div class="metric-label">Reps</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{acc:.1f}%</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{fps:.0f}</div>
                            <div class="metric-label">FPS</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{int(duration//60)}:{int(duration%60):02d}</div>
                            <div class="metric-label">Duration</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Accuracy progress bar ─────────────────────────────────────────
                progress_ph.progress(min(acc / 100, 1.0))

                # ── Intelligent feedback display ──────────────────────────────────
                with feedback_ph.container():
                    # Phase / direction badge
                    dir_sym   = {"up": "↑ Ascending", "down": "↓ Descending"}.get(system.direction, "— Hold")
                    phase_lbl = system.phase.upper()
                    st.markdown(f"""
                    <div style="margin-bottom:4px;">
                        <span class="phase-badge">{dir_sym}</span>
                        <span class="phase-badge">{phase_lbl}</span>
                        <span style="color:#475569; font-size:12px;">Angle: <strong style="color:#93c5fd">{int(system.prev_angle or 0)}°</strong></span>
                    </div>
                    """, unsafe_allow_html=True)
                    render_feedback(feedback_text)

                # ── Camera frame ──────────────────────────────────────────────────
                rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                frame_ph.image(rgb, channels="RGB", use_container_width=True)

            cap.release()
            st.session_state.running = False

    # ── Post-session summary ─────────────────────────────────────────────────
    if not st.session_state.running and st.session_state.system:
        system   = st.session_state.system
        acc      = (system.correct_frames / system.total_frames * 100) if system.total_frames else 0
        duration = time.time() - system.session_start

        st.balloons()
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-title">🎉 Session Complete!</div>
            <div class="summary-sub">Nice work, <strong>{st.session_state.user_name}</strong></div>
            <div class="metric-row" style="margin-top:20px; justify-content:center;">
                <div class="metric-card">
                    <div class="metric-value">{system.rep_count}</div>
                    <div class="metric-label">Reps Done</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{acc:.1f}%</div>
                    <div class="metric-label">Form Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{int(duration//60)}:{int(duration%60):02d}</div>
                    <div class="metric-label">Duration</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔁  Train Again", use_container_width=True):
                st.session_state.system  = RehabSystem(exercise=st.session_state.selected_exercise)
                st.session_state.running = True
                st.rerun()
        with col2:
            if st.button("🏠  Start Over", use_container_width=True):
                for k, v in defaults.items():
                    st.session_state[k] = v
                st.rerun()
