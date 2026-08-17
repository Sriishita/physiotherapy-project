"""
voice_feedback.py — macOS built-in TTS. No pip installs needed.
"""
import threading
import time
import subprocess

last_feedback    = ""
last_spoken_time = 0.0
SPEAK_DELAY      = 2.5
_tts_lock        = threading.Lock()


def _speak_macos(text: str):
    with _tts_lock:
        try:
            subprocess.run(["say", "-r", "175", "-v", "Samantha", text],
                           check=False, timeout=6)
        except Exception as e:
            print(f"[Voice] {e}")


def speak(text: str):
    global last_feedback, last_spoken_time
    if not text:
        return
    now = time.time()
    if text == last_feedback or (now - last_spoken_time) < SPEAK_DELAY:
        return
    last_feedback    = text
    last_spoken_time = now
    threading.Thread(target=_speak_macos, args=(text,), daemon=True).start()