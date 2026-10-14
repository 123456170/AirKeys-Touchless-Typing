"""
AirKeys: Touchless Typing — single-file Streamlit app.

Privacy:
- All processing is local.
- No API key is required.
- No video or audio is uploaded.
- log.csv is written only when the user enables Record.

License summary for required open-source libraries:
- Streamlit: Apache-2.0
- OpenCV: Apache-2.0
- MediaPipe: Apache-2.0
- NumPy: BSD-3-Clause
- pandas: BSD-3-Clause

Assumptions:
- The user requested only requirements.txt and app.py, so config, tests, keyboard,
  prediction, selection, demo simulation, and UI are embedded in app.py.
- MediaPipe Tasks model files are optional local binary assets. If absent, the app
  falls back to local MediaPipe legacy tracking when available, and always provides
  a synthetic demo so the dashboard is never blank.
"""

import os
import re
import csv
import math
import time
import random
import threading
from collections import deque, Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None


BACKSPACE = "\b"
ENTER = "\n"

# Starting thresholds. Tune these on real recordings.
CONFIG = {
    "dwell_ms": 700, # default dwell time in milliseconds
    "pinch_on": 0.35, # pinch ratio threshold to start pinch
    "pinch_off": 0.55, # pinch ratio threshold to release pinch
    "magnet_radius_palm": 0.55, # magnetic key capture radius multiplied by palm width
    "hysteresis_radius_mult": 1.25, # leave radius multiplier for key hover hysteresis
    "dead_zone_palm": 0.045, # fingertip dead zone multiplied by palm width
    "pinch_move_tolerance_palm": 0.18, # allowed movement during pinch multiplied by palm width
    "one_euro_min_cutoff": 1.2, # One Euro filter minimum cutoff in Hz
    "one_euro_beta": 0.08, # One Euro filter speed coefficient
    "one_euro_d_cutoff": 1.0, # One Euro filter derivative cutoff in Hz
    "arming_hold_ms": 1000, # open palm hold time required to arm risky edits
    "arming_ttl_ms": 5000, # how long arming remains valid in milliseconds
    "edit_hold_ms": 350, # fist hold time required for backspace in milliseconds
    "space_hold_ms": 300, # space gesture hold time in milliseconds
    "refractory_ms": 250, # minimum time between repeated actions in milliseconds
    "adaptive_key_scale": 1.35, # size multiplier for predicted keys
    "camera_index": 0, # webcam device index
    "hand_model_path": "hand_landmarker.task", # optional MediaPipe Tasks hand model path
    "gesture_model_path": "gesture_recognizer.task", # optional MediaPipe Tasks gesture model path
    "log_path": "log.csv", # session log CSV path
    "max_suggestions": 3, # maximum word suggestions shown
    "fps": 15, # demo and render target frames per second
    "live_seconds": 12, # default live camera session length in seconds
    "relative_gain_x": 5.0, # horizontal gain from palm-relative fingertip to keyboard
    "relative_gain_y": 6.0, # vertical gain from palm-relative fingertip to keyboard
    "relative_center_x": 0.5, # horizontal keyboard center for neutral palm pose
    "relative_center_y": 0.62, # vertical keyboard center for neutral palm pose
    "open_ext_min": 1.05, # minimum normalized fingertip extension for open palm
    "open_pinch_min": 0.65, # minimum pinch ratio for open palm
    "fist_curl_max": 0.65, # maximum tip-to-MCP curl for fist
    "fist_pinch_max": 0.45, # maximum pinch ratio for fist
    "space_ext_min": 1.0, # minimum extension for index and middle in space gesture
    "space_curl_max": 0.75, # maximum curl for ring and pinky in space gesture
}

TEST_PHRASES = [
    "the quick brown fox",
    "jumped over the lazy dog",
    "touchless typing is interesting",
    "keep your hand relaxed",
    "pinch to select a key",
    "dwell to type one letter",
    "suggestions reduce movement",
    "error rate matters too",
    "local processing protects privacy",
    "practice makes smoother gestures",
]

CORPUS = TEST_PHRASES + [
    "hello world",
    "this is a demo",
    "the best guess",
    "air keys touchless keyboard",
    "media pipe hand tracking",
    "streamlit local app",
    "type with gestures",
    "open palm arm backspace",
    "peace sign space",
    "finger pinch release",
]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def display_key(ch: Optional[str]) -> str:
    if ch is None:
        return "—"
    if ch == BACKSPACE:
        return "⌫"
    if ch == ENTER:
        return "⏎"
    if ch == " ":
        return "␣"
    return ch.upper()


@dataclass
class GestureEvent:
    type: str
    hand: str
    t_ms: int
    confidence: float
    params: Dict[str, Any] = field(default_factory=dict)


def _alpha(rc: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(rc, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class LowPassFilter:
    def __init__(self) -> None:
        self.initialized = False
        self.y = 0.0

    def filter(self, x: float, a: float) -> float:
        if not self.initialized:
            self.y = x
            self.initialized = True
        else:
            self.y = a * x + (1.0 - a) * self.y
        return self.y


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_lp = LowPassFilter()
        self.dx_lp = LowPassFilter()
        self.t_prev: Optional[float] = None
        self.x_prev: Optional[float] = None

    def filter(self, x: float, t: float) -> float:
        if self.t_prev is None:
            dt = 1.0 / 30.0
        else:
            dt = max(1e-6, t - self.t_prev)

        if self.x_prev is None:
            dx = 0.0
        else:
            dx = (x - self.x_prev) / dt

        edx = self.dx_lp.filter(dx, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = _alpha(cutoff, dt)
        y = self.x_lp.filter(x, a)

        self.t_prev = t
        self.x_prev = x
        return y


class PointOneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def filter(self, point: Tuple[float, float], t: float) -> Tuple[float, float]:
        return (self.fx.filter(point[0], t), self.fy.filter(point[1], t))


class Key:
    __slots__ = ("char", "label", "x", "y", "w", "h")

    def __init__(self, char: str, label: str, x: float, y: float, w: float, h: float) -> None:
        self.char = char
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


class Keyboard:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.keys: List[Key] = []
        self.by_char: Dict[str, Key] = {}
        self._build()

    def _add(self, char: str, label: str, x: float, y: float, w: float, h: float) -> None:
        k = Key(char, label, x, y, w, h)
        self.keys.append(k)
        self.by_char[char] = k

    def _build(self) -> None:
        rows = [
            ("qwertyuiop", 0.08),
            ("asdfghjkl;", 0.28),
            ("zxcvbnm,./", 0.48),
        ]
        kw = 0.084
        kh = 0.155

        for row, y in rows:
            n = len(row)
            total = n * kw
            x0 = (1.0 - total) / 2.0
            for i, ch in enumerate(row):
                self._add(ch, ch.upper(), x0 + i * kw, y, kw, kh)

        self._add(BACKSPACE, "⌫", 0.08, 0.70, 0.16, 0.16)
        self._add(" ", "SPACE", 0.28, 0.70, 0.44, 0.16)
        self._add(ENTER, "⏎", 0.76, 0.70, 0.16, 0.16)

    def center(self, char: str) -> Optional[Tuple[float, float]]:
        k = self.by_char.get(char)
        return k.center() if k else None

    def _inside(self, k: Key, point: Tuple[float, float], scale: float) -> bool:
        cx, cy = k.center()
        hw = k.w * scale / 2.0
        hh = k.h * scale / 2.0
        return abs(point[0] - cx) <= hw and abs(point[1] - cy) <= hh

    def hit_test(
        self,
        point: Optional[Tuple[float, float]],
        palm_width: float,
        current_key: Optional[str] = None,
        predictions: Optional[List[str]] = None,
    ) -> Optional[str]:
        if point is None:
            return None

        pw = max(float(palm_width), 0.04)
        enter = self.cfg["magnet_radius_palm"] * pw
        leave = enter * self.cfg["hysteresis_radius_mult"]
        pred_set = set(predictions or [])

        def scale_for(k: Key) -> float:
            return self.cfg["adaptive_key_scale"] if k.char in pred_set else 1.0

        if current_key is not None and current_key in self.by_char:
            k = self.by_char[current_key]
            c = k.center()
            d = dist(point, c)
            if self._inside(k, point, scale_for(k)) or d < leave:
                return k.char

        inside: List[Tuple[float, Key]] = []
        best: Optional[Key] = None
        best_d = 1e9

        for k in self.keys:
            c = k.center()
            d = dist(point, c)
            if d < best_d:
                best_d = d
                best = k
            if self._inside(k, point, scale_for(k)):
                inside.append((d, k))

        if inside:
            return min(inside, key=lambda item: item[0])[1].char

        if best is not None and best_d < enter:
            return best.char

        return None


class NGramPredictor:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.word_freq: Counter = Counter()
        self.word_bigram: Dict[str, Counter] = defaultdict(Counter)
        self.char_bigram: Dict[str, Counter] = defaultdict(Counter)
        self._build()

    def _tokens(self, text: str) -> List[str]:
        return re.findall(r"[a-z']+", text.lower())

    def _build(self) -> None:
        for phrase in CORPUS:
            words = self._tokens(phrase)
            self.word_freq.update(words)
            for i in range(len(words) - 1):
                self.word_bigram[words[i]][words[i + 1]] += 1

        for word, freq in self.word_freq.items():
            for a, b in zip(word, word[1:]):
                self.char_bigram[a][b] += freq

    def suggestions(self, text: str) -> List[str]:
        text = text or ""
        if not text:
            return []

        candidates: List[Tuple[str, float]] = []

        if text.endswith(" "):
            parts = text.strip().split()
            prev = parts[-1].lower() if parts else ""
            counts = self.word_bigram.get(prev, Counter())
            for w, c in counts.items():
                candidates.append((w, c * 2.0 + self.word_freq.get(w, 0) * 0.1))
            if not candidates:
                candidates = [(w, float(f)) for w, f in self.word_freq.most_common(30)]
        else:
            partial = text.split()[-1].lower() if text else ""
            if not partial:
                return []
            for w, f in self.word_freq.items():
                if w.startswith(partial) and w != partial:
                    candidates.append((w, f + len(partial) * 5.0))

        candidates.sort(key=lambda item: item[1], reverse=True)
        seen = set()
        out: List[str] = []
        for w, _ in candidates:
            if w not in seen:
                seen.add(w)
                out.append(w)
            if len(out) >= self.cfg["max_suggestions"]:
                break
        return out

    def predicted_letters(self, text: str) -> List[str]:
        text = text or ""
        letters: Counter = Counter()

        if text.endswith(" "):
            for w in self.suggestions(text):
                if w:
                    letters[w[0]] += self.word_freq.get(w, 1)
        else:
            partial = text.split()[-1].lower() if text else ""
            if partial:
                for w, f in self.word_freq.items():
                    if w.startswith(partial) and len(w) > len(partial):
                        letters[w[len(partial)]] += f
            if not letters and text:
                last = text[-1].lower()
                letters.update(self.char_bigram.get(last, Counter()))

        return [c for c, _ in letters.most_common(6) if c.isalpha()]


class TypingTest:
    def __init__(self, target: str) -> None:
        self.target = target
        self.pos = 0
        self.typed = 0
        self.errors = 0
        self.start: Optional[int] = None
        self.end: Optional[int] = None
        self.log: List[Tuple[int, str, str, bool]] = []

    def feed(self, ch: str, t_ms: int, method: str) -> bool:
        if self.start is None:
            self.start = t_ms
        self.end = t_ms
        self.typed += 1

        if ch == BACKSPACE:
            if self.pos > 0:
                self.pos -= 1
            correct = True
        else:
            expected = self.target[self.pos] if self.pos < len(self.target) else None
            correct = ch == expected
            if correct:
                self.pos += 1
            else:
                self.errors += 1

        self.log.append((t_ms, ch, method, correct))
        return correct

    def wpm(self) -> float:
        if self.start is None or self.end is None or self.end <= self.start:
            return 0.0
        minutes = (self.end - self.start) / 60000.0
        if minutes <= 0:
            return 0.0
        return (self.typed / 5.0) / minutes

    def error_rate(self) -> float:
        return self.errors / self.typed if self.typed else 0.0


class SelectionEngine:
    def __init__(self, cfg: Dict[str, Any], keyboard: Keyboard) -> None:
        self.cfg = cfg
        self.kb = keyboard
        self.filt = PointOneEuroFilter(
            cfg["one_euro_min_cutoff"],
            cfg["one_euro_beta"],
            cfg["one_euro_d_cutoff"],
        )

        self.stable_point: Optional[Tuple[float, float]] = None
        self.current_key: Optional[str] = None

        self.dwell_key: Optional[str] = None
        self.dwell_elapsed = 0.0
        self.progress = 0.0
        self.status = "Move your index finger over the keyboard."

        self.pinch_raw = False
        self.prev_pinch_raw = False
        self.pinch_active = False
        self.pinch_key: Optional[str] = None
        self.pinch_start_point: Optional[Tuple[float, float]] = None
        self.pinch_cancelled = False

        self.open_hold = 0.0
        self.fist_hold = 0.0
        self.space_hold = 0.0

        self.armed = False
        self.armed_expiry = 0

        self.last_t: Optional[int] = None
        self.last_action = -10**9
        self.last_space = -10**9
        self.last_backspace = -10**9

    def update(
        self,
        point: Optional[Tuple[float, float]],
        t_ms: int,
        palm_width: float,
        raw_gestures: Optional[Dict[str, Any]] = None,
        predictions: Optional[List[str]] = None,
        hand: str = "right",
    ) -> List[GestureEvent]:
        raw = raw_gestures or {}
        events: List[GestureEvent] = []

        if self.last_t is None:
            dt_ms = 0.0
        else:
            dt_ms = max(0.0, min(100.0, float(t_ms - self.last_t)))
        self.last_t = t_ms

        if point is None:
            if self.pinch_active:
                events.append(GestureEvent("pinch_cancel", hand, t_ms, 0.7, {}))
                self.pinch_active = False
                self.pinch_key = None
                self.pinch_start_point = None
                self.pinch_cancelled = False

            self.current_key = None
            self.dwell_key = None
            self.dwell_elapsed = 0.0
            self.progress = 0.0
            self.prev_pinch_raw = False
            self.status = "No hand detected."
            return events

        filtered = self.filt.filter(point, t_ms / 1000.0)
        pw = max(float(palm_width or 0.12), 0.04)

        if self.stable_point is None or dist(filtered, self.stable_point) >= self.cfg["dead_zone_palm"] * pw:
            self.stable_point = filtered

        hover_point = self.stable_point

        ratio = raw.get("pinch_ratio")
        if ratio is not None:
            if not self.pinch_raw and float(ratio) < self.cfg["pinch_on"]:
                self.pinch_raw = True
            elif self.pinch_raw and float(ratio) > self.cfg["pinch_off"]:
                self.pinch_raw = False
        else:
            self.pinch_raw = bool(raw.get("pinch", False))

        key = self.kb.hit_test(hover_point, pw, self.current_key, predictions)
        self.current_key = key

        if self.pinch_active and self.pinch_start_point is not None:
            if key != self.pinch_key or dist(hover_point, self.pinch_start_point) > self.cfg["pinch_move_tolerance_palm"] * pw:
                if not self.pinch_cancelled:
                    self.pinch_cancelled = True
                    self.status = "Pinch cancelled: hand moved."

        if self.pinch_raw and not self.prev_pinch_raw:
            if key:
                self.pinch_active = True
                self.pinch_key = key
                self.pinch_start_point = hover_point
                self.pinch_cancelled = False
                events.append(GestureEvent("pinch_start", hand, t_ms, 0.9, {"key": key}))
                self.status = f"Pinch started on {display_key(key)}"
            else:
                self.status = "Pinch must start over a key."

        elif not self.pinch_raw and self.prev_pinch_raw:
            if self.pinch_active:
                if (
                    not self.pinch_cancelled
                    and key == self.pinch_key
                    and t_ms - self.last_action >= self.cfg["refractory_ms"]
                ):
                    events.append(GestureEvent("pinch_type", hand, t_ms, 0.95, {"key": key}))
                    self.last_action = t_ms
                    self.status = f"Pinch typed {display_key(key)}"
                else:
                    events.append(GestureEvent("pinch_cancel", hand, t_ms, 0.8, {}))
                    self.status = "Pinch cancelled."

                self.pinch_active = False
                self.pinch_key = None
                self.pinch_start_point = None
                self.pinch_cancelled = False

        gesture_busy = bool(raw.get("open_palm") or raw.get("fist") or raw.get("space"))

        if not self.pinch_raw and not self.pinch_active and not gesture_busy:
            if key and key == self.dwell_key:
                self.dwell_elapsed += dt_ms
            elif key:
                self.dwell_key = key
                self.dwell_elapsed = 0.0
            else:
                self.dwell_key = None
                self.dwell_elapsed = 0.0

            if (
                self.dwell_key
                and self.dwell_elapsed >= self.cfg["dwell_ms"]
                and t_ms - self.last_action >= self.cfg["refractory_ms"]
            ):
                ch = self.dwell_key
                events.append(GestureEvent("dwell_type", hand, t_ms, 0.9, {"key": ch}))
                self.last_action = t_ms
                self.dwell_elapsed = 0.0
                self.status = f"Dwell typed {display_key(ch)}"
        else:
            self.dwell_key = None
            self.dwell_elapsed = 0.0

        if self.dwell_key and not self.pinch_active and not self.pinch_raw and not gesture_busy:
            self.progress = clamp(self.dwell_elapsed / max(1.0, float(self.cfg["dwell_ms"])), 0.0, 1.0)
        else:
            self.progress = 0.0

        if raw.get("open_palm"):
            self.open_hold += dt_ms
            if self.open_hold >= self.cfg["arming_hold_ms"] and not self.armed:
                self.armed = True
                self.armed_expiry = t_ms + self.cfg["arming_ttl_ms"]
                events.append(GestureEvent("arming", hand, t_ms, 0.85, {}))
                self.status = "Armed: fist will send backspace."
        else:
            self.open_hold = 0.0

        if raw.get("fist"):
            if self.armed and t_ms < self.armed_expiry:
                self.fist_hold += dt_ms
                if self.fist_hold >= self.cfg["edit_hold_ms"] and t_ms - self.last_backspace >= self.cfg["refractory_ms"]:
                    events.append(GestureEvent("backspace", hand, t_ms, 0.9, {}))
                    self.last_backspace = t_ms
                    self.armed = False
                    self.fist_hold = 0.0
                    self.status = "Backspace sent."
            else:
                self.fist_hold = 0.0
                if not self.armed:
                    self.status = "Open palm for 1 second to arm backspace."
        else:
            self.fist_hold = 0.0

        if raw.get("space"):
            self.space_hold += dt_ms
            if self.space_hold >= self.cfg["space_hold_ms"] and t_ms - self.last_space >= self.cfg["refractory_ms"]:
                events.append(GestureEvent("space", hand, t_ms, 0.9, {}))
                self.last_space = t_ms
                self.space_hold = 0.0
                self.status = "Space sent."
        else:
            self.space_hold = 0.0

        if not events and key and not self.pinch_active:
            self.status = f"Hovering {display_key(key)}"

        self.prev_pinch_raw = self.pinch_raw
        return events


def classify_raw_gestures(landmarks: List[Tuple[float, float]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not landmarks or len(landmarks) != 21:
        return {}

    arr = np.asarray(landmarks, dtype=float)
    palm_width = max(float(np.linalg.norm(arr[5] - arr[17])), 1e-4)
    palm_center = arr[[0, 5, 9, 13, 17]].mean(axis=0)

    pinch_ratio = float(np.linalg.norm(arr[4] - arr[8]) / palm_width)

    ext: Dict[int, float] = {}
    curl: Dict[int, float] = {}
    for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        ext[tip] = float(np.linalg.norm(arr[tip] - palm_center) / palm_width)
        curl[tip] = float(np.linalg.norm(arr[tip] - arr[mcp]) / palm_width)

    open_palm = (
        all(ext[t] > cfg["open_ext_min"] for t in (8, 12, 16, 20))
        and pinch_ratio > cfg["open_pinch_min"]
    )

    fist = (
        all(curl[t] < cfg["fist_curl_max"] for t in (8, 12, 16, 20))
        and pinch_ratio < cfg["fist_pinch_max"]
    )

    space = (
        ext[8] > cfg["space_ext_min"]
        and ext[12] > cfg["space_ext_min"]
        and curl[16] < cfg["space_curl_max"]
        and curl[20] < cfg["space_curl_max"]
    )

    return {
        "pinch_ratio": pinch_ratio,
        "open_palm": bool(open_palm),
        "fist": bool(fist),
        "space": bool(space),
        "palm_width": palm_width,
        "palm_center": palm_center.tolist(),
    }


def landmarks_to_keyboard_point(
    landmarks: List[Tuple[float, float]],
    cfg: Dict[str, Any],
) -> Tuple[Tuple[float, float], float]:
    arr = np.asarray(landmarks, dtype=float)
    palm_width = max(float(np.linalg.norm(arr[5] - arr[17])), 1e-4)
    palm_center = arr[[0, 5, 9, 13, 17]].mean(axis=0)
    index_tip = arr[8]

    x = float(index_tip[0] - palm_center[0]) * cfg["relative_gain_x"] + cfg["relative_center_x"]
    y = float(index_tip[1] - palm_center[1]) * cfg["relative_gain_y"] + cfg["relative_center_y"]

    return (clamp(x, 0.02, 0.98), clamp(y, 0.02, 0.98)), palm_width


class NullHandDetector:
    backend = "none"

    def detect(self, frame_rgb: np.ndarray) -> List[List[Tuple[float, float]]]:
        return []

    def close(self) -> None:
        return None


class LegacyHandDetector:
    backend = "mediapipe-legacy"

    def __init__(self) -> None:
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame_rgb: np.ndarray) -> List[List[Tuple[float, float]]]:
        result = self.hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            return [[(lm.x, lm.y) for lm in hand.landmark] for hand in result.multi_hand_landmarks]
        return []

    def close(self) -> None:
        self.hands.close()


class TasksHandDetector:
    backend = "mediapipe-tasks"

    def __init__(self, model_path: str) -> None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray) -> List[List[Tuple[float, float]]]:
        rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        if result.hand_landmarks:
            return [[(lm.x, lm.y) for lm in hand] for hand in result.hand_landmarks]
        return []

    def close(self) -> None:
        self.detector.close()


def get_hand_detector(cfg: Dict[str, Any]):
    if mp is None:
        return NullHandDetector()

    if os.path.exists(cfg["hand_model_path"]):
        try:
            return TasksHandDetector(cfg["hand_model_path"])
        except Exception:
            pass

    try:
        return LegacyHandDetector()
    except Exception:
        return NullHandDetector()


class FrameGrabber:
    def __init__(self, camera_index: int) -> None:
        self.ok = False
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.cap = None

        if cv2 is None:
            return

        self.cap = cv2.VideoCapture(camera_index)
        self.ok = self.cap.isOpened()

        if self.ok:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Optional[np.ndarray]:
        if not self.ok:
            return None
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def release(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.ok and self.cap is not None:
            self.cap.release()


def append_log_row(row: Tuple[int, str, str, bool], record: bool) -> None:
    if not record:
        return

    path = CONFIG["log_path"]
    exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["t_ms", "key", "method", "correct"])
        writer.writerow(row)


class LivePipeline:
    def __init__(
        self,
        cfg: Dict[str, Any],
        keyboard: Keyboard,
        predictor: NGramPredictor,
        selection: SelectionEngine,
        initial_text: str,
        record: bool,
        test_target: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.kb = keyboard
        self.pred = predictor
        self.sel = selection
        self.record = record
        self.text = initial_text
        self.logs: List[Tuple[int, str, str, bool]] = []
        self.test = TypingTest(test_target) if test_target else None

        self.grabber = FrameGrabber(cfg["camera_index"]) if cv2 is not None else None
        self.detector = get_hand_detector(cfg)

        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        self.state: Dict[str, Any] = {
            "frame": None,
            "point": None,
            "palm_width": 0.15,
            "hover": None,
            "progress": 0.0,
            "text": self.text,
            "status": "Starting local camera pipeline...",
            "suggestions": [],
            "predictions": [],
            "armed": False,
            "t_ms": 0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "log_count": 0,
            "last_log": None,
            "test_wpm": 0.0,
            "test_error_rate": 0.0,
            "test_pos": 0,
        }

    def start(self) -> None:
        if self.grabber is not None and self.grabber.ok:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        else:
            with self.lock:
                self.state["status"] = "Camera unavailable."

    def _log(self, t_ms: int, key: str, method: str, correct: bool) -> None:
        row = (t_ms, key, method, correct)
        self.logs.append(row)
        append_log_row(row, self.record)

    def _apply_events(self, events: List[GestureEvent]) -> None:
        for ev in events:
            if ev.type in ("dwell_type", "pinch_type"):
                ch = ev.params.get("key", "")
                self.text += ch
                method = "dwell" if ev.type == "dwell_type" else "pinch"
                correct = self.test.feed(ch, ev.t_ms, method) if self.test else True
                self._log(ev.t_ms, ch, method, correct)

            elif ev.type == "space":
                self.text += " "
                correct = self.test.feed(" ", ev.t_ms, "gesture") if self.test else True
                self._log(ev.t_ms, "SPACE", "gesture", correct)

            elif ev.type == "backspace":
                if self.text:
                    self.text = self.text[:-1]
                correct = self.test.feed(BACKSPACE, ev.t_ms, "gesture") if self.test else True
                self._log(ev.t_ms, "BACKSPACE", "gesture", correct)

    def _run(self) -> None:
        fps_q: deque = deque(maxlen=30)
        last = time.time()

        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            frame = self.grabber.read() if self.grabber is not None else None

            if frame is None:
                time.sleep(0.02)
                continue

            t_ms = int(time.time() * 1000)
            extra_status = ""

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = self.detector.detect(rgb)
            except Exception as exc:
                hands = []
                extra_status = f" Detector error: {exc}"

            raw: Dict[str, Any] = {}
            point: Optional[Tuple[float, float]] = None
            palm_width = 0.15

            if hands:
                landmarks = hands[0]
                raw = classify_raw_gestures(landmarks, self.cfg)
                point, palm_width = landmarks_to_keyboard_point(landmarks, self.cfg)

            predictions = self.pred.predicted_letters(self.text)

            if point is not None:
                events = self.sel.update(point, t_ms, palm_width, raw, predictions)
            else:
                events = self.sel.update(None, t_ms, palm_width, {}, predictions)

            self._apply_events(events)

            now = time.time()
            fps_q.append(1.0 / max(now - last, 1e-3))
            last = now

            latency_ms = (time.perf_counter() - t0) * 1000.0
            avg_fps = float(sum(fps_q) / len(fps_q)) if fps_q else 0.0

            with self.lock:
                self.state.update(
                    {
                        "frame": frame,
                        "point": point,
                        "palm_width": palm_width,
                        "hover": self.sel.current_key,
                        "progress": self.sel.progress,
                        "text": self.text,
                        "status": self.sel.status + extra_status,
                        "suggestions": self.pred.suggestions(self.text),
                        "predictions": predictions,
                        "armed": self.sel.armed,
                        "t_ms": t_ms,
                        "fps": avg_fps,
                        "latency_ms": latency_ms,
                        "log_count": len(self.logs),
                        "last_log": self.logs[-1] if self.logs else None,
                        "test_wpm": self.test.wpm() if self.test else 0.0,
                        "test_error_rate": self.test.error_rate() if self.test else 0.0,
                        "test_pos": self.test.pos if self.test else 0,
                    }
                )

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def get_logs(self) -> List[Tuple[int, str, str, bool]]:
        return list(self.logs)

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.5)
        if self.grabber is not None:
            self.grabber.release()
        if hasattr(self.detector, "close"):
            try:
                self.detector.close()
            except Exception:
                pass


class DemoSimulator:
    def __init__(self, cfg: Dict[str, Any], keyboard: Keyboard, predictor: NGramPredictor) -> None:
        self.cfg = cfg
        self.kb = keyboard
        self.pred = predictor
        self.sel = SelectionEngine(cfg, keyboard)
        self.text = ""
        self.frames: List[Dict[str, Any]] = []
        self.logs: List[Tuple[int, str, str, bool]] = []
        self.target = ""
        self.test: Optional[TypingTest] = None
        self.t_ms = 0
        self.dt = 1000.0 / float(cfg["fps"])

    def set_target(self, target: str) -> None:
        self.target = target
        self.test = TypingTest(target)

    def _apply_events(self, events: List[GestureEvent]) -> None:
        for ev in events:
            if ev.type in ("dwell_type", "pinch_type"):
                ch = ev.params.get("key", "")
                self.text += ch
                method = "dwell" if ev.type == "dwell_type" else "pinch"
                correct = self.test.feed(ch, ev.t_ms, method) if self.test else True
                self.logs.append((ev.t_ms, ch, method, correct))

            elif ev.type == "space":
                self.text += " "
                correct = self.test.feed(" ", ev.t_ms, "gesture") if self.test else True
                self.logs.append((ev.t_ms, "SPACE", "gesture", correct))

            elif ev.type == "backspace":
                if self.text:
                    self.text = self.text[:-1]
                correct = self.test.feed(BACKSPACE, ev.t_ms, "gesture") if self.test else True
                self.logs.append((ev.t_ms, "BACKSPACE", "gesture", correct))

    def _emit_frame(
        self,
        point: Optional[Tuple[float, float]],
        palm_width: float,
        raw: Dict[str, Any],
        force_status: Optional[str] = None,
    ) -> None:
        predictions = self.pred.predicted_letters(self.text)
        events = self.sel.update(point, int(self.t_ms), palm_width, raw, predictions)
        self._apply_events(events)

        frame = {
            "t_ms": int(self.t_ms),
            "point": point,
            "palm_width": palm_width,
            "hover": self.sel.current_key,
            "progress": self.sel.progress,
            "text": self.text,
            "status": force_status or self.sel.status,
            "suggestions": self.pred.suggestions(self.text),
            "predictions": predictions,
            "armed": self.sel.armed,
        }
        self.frames.append(frame)
        self.t_ms += int(self.dt)

    def move_to(
        self,
        target: Optional[Tuple[float, float]],
        frames: int = 5,
        palm_width: float = 0.15,
        raw: Optional[Dict[str, Any]] = None,
    ) -> None:
        if target is None:
            return

        start = self.frames[-1]["point"] if self.frames else (0.5, 0.85)
        if start is None:
            start = (0.5, 0.85)

        raw = raw or {}

        for i in range(max(1, frames)):
            u = (i + 1) / float(max(1, frames))
            ease = u * u * (3.0 - 2.0 * u)
            x = start[0] + (target[0] - start[0]) * ease
            y = start[1] + (target[1] - start[1]) * ease

            noise = 0.0015 * math.sin(self.t_ms / 70.0 + i)
            point = (clamp(x + noise, 0.02, 0.98), clamp(y + noise * 0.7, 0.02, 0.98))
            self._emit_frame(point, palm_width, raw)

    def hold(
        self,
        point: Tuple[float, float],
        duration_ms: float,
        palm_width: float = 0.15,
        raw: Optional[Dict[str, Any]] = None,
    ) -> None:
        raw = raw or {}
        n = max(1, int(duration_ms / self.dt))
        for _ in range(n):
            self._emit_frame(point, palm_width, raw)

    def pinch_at(self, point: Tuple[float, float], palm_width: float = 0.15, invalid: bool = False) -> None:
        self._emit_frame(point, palm_width, {"pinch_ratio": 0.25})
        self.hold(point, 120.0, palm_width, {"pinch_ratio": 0.25})

        if invalid:
            away = (clamp(point[0] + 0.22, 0.02, 0.98), clamp(point[1] - 0.10, 0.02, 0.98))
            self.move_to(away, 4, palm_width, {"pinch_ratio": 0.25})
            self._emit_frame(away, palm_width, {"pinch_ratio": 0.75})
            self.move_to(point, 4, palm_width, {})

        self._emit_frame(point, palm_width, {"pinch_ratio": 0.75})

    def space_gesture(self, point: Tuple[float, float] = (0.5, 0.88), palm_width: float = 0.15) -> None:
        self.move_to(point, 3, palm_width, {})
        self.hold(point, self.cfg["space_hold_ms"] + 80.0, palm_width, {"space": True})
        self._emit_frame(point, palm_width, {"space": False})

    def arm_and_backspace(self, point: Tuple[float, float] = (0.5, 0.88), palm_width: float = 0.15) -> None:
        self.move_to(point, 3, palm_width, {})
        self.hold(point, self.cfg["arming_hold_ms"] + 80.0, palm_width, {"open_palm": True})
        self._emit_frame(point, palm_width, {"open_palm": False})
        self.hold(point, self.cfg["edit_hold_ms"] + 80.0, palm_width, {"fist": True})
        self._emit_frame(point, palm_width, {"fist": False})

    def type_sequence(self, text: str, method: str, invalid_first_pinch: bool = False) -> None:
        first_non_space = True

        for ch in text:
            if ch == " ":
                self.space_gesture()
                continue

            center = self.kb.center(ch)
            if center is None:
                continue

            self.move_to(center, 5, 0.15, {})

            if method == "dwell":
                self.hold(center, self.cfg["dwell_ms"] + 120.0, 0.15, {})
            else:
                invalid = invalid_first_pinch and first_non_space
                self.pinch_at(center, 0.15, invalid=invalid)
                first_non_space = False


def summarize_logs(logs: List[Tuple[int, str, str, bool]], start_ms: int, end_ms: int) -> Dict[str, Any]:
    typed_rows = [row for row in logs if row[1] != "BACKSPACE"]
    chars = len(typed_rows)
    errors = sum(1 for row in typed_rows if not row[3])
    duration_ms = max(1, end_ms - start_ms)
    wpm = (chars / 5.0) / (duration_ms / 60000.0)
    error_rate = errors / chars if chars else 0.0

    return {
        "characters": chars,
        "errors": errors,
        "wpm": round(wpm, 2),
        "error_rate": round(error_rate, 4),
        "duration_s": round(duration_ms / 1000.0, 1),
    }


def generate_demo_frames(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, List[Tuple[int, str, str, bool]], Dict[str, Any]]:
    kb = Keyboard(cfg)
    pred = NGramPredictor(cfg)
    sim = DemoSimulator(cfg, kb, pred)

    sim.set_target("hello world")
    sim.type_sequence("hello worldx", "dwell")
    sim.arm_and_backspace()

    sim.set_target(" ok")
    sim.type_sequence(" ok", "pinch", invalid_first_pinch=True)

    start_ms = sim.frames[0]["t_ms"] if sim.frames else 0
    end_ms = sim.frames[-1]["t_ms"] if sim.frames else 0
    report = summarize_logs(sim.logs, start_ms, end_ms)

    return sim.frames, sim.text, sim.logs, report


def generate_study(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(2026)
    phrases = TEST_PHRASES[:10]
    methods = ["Dwell", "Pinch", "Dwell + suggestions", "Pinch + suggestions"]

    base_wpm = {
        "Dwell": 10.0,
        "Pinch": 12.5,
        "Dwell + suggestions": 12.0,
        "Pinch + suggestions": 15.0,
    }

    base_err = {
        "Dwell": 0.09,
        "Pinch": 0.07,
        "Dwell + suggestions": 0.06,
        "Pinch + suggestions": 0.05,
    }

    rows: List[Dict[str, Any]] = []

    for user in range(1, 6):
        for method in methods:
            for phrase in phrases:
                chars = len(phrase)
                wpm = max(2.0, rng.gauss(base_wpm[method], 1.8))
                err = min(0.45, max(0.0, rng.gauss(base_err[method], 0.025)))
                incorrect = int(round(err * chars))
                duration_ms = int(((chars / 5.0) / wpm) * 60000.0)

                rows.append(
                    {
                        "user": f"U{user}",
                        "method": method,
                        "phrase": phrase,
                        "characters": chars,
                        "wpm": round(wpm, 2),
                        "error_rate": round(incorrect / chars, 4),
                        "incorrect_chars": incorrect,
                        "duration_ms": duration_ms,
                    }
                )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("method")
        .agg(
            mean_wpm=("wpm", "mean"),
            sd_wpm=("wpm", "std"),
            mean_error=("error_rate", "mean"),
            sd_error=("error_rate", "std"),
            trials=("wpm", "count"),
        )
        .reset_index()
        .round(3)
    )

    return df, summary


def make_synthetic_landmarks(kind: str = "open") -> List[Tuple[float, float]]:
    arr = np.zeros((21, 2), dtype=float)

    arr[0] = (0.50, 0.80)
    arr[5] = (0.35, 0.55)
    arr[9] = (0.45, 0.52)
    arr[13] = (0.55, 0.52)
    arr[17] = (0.65, 0.55)

    arr[4] = (0.25, 0.45)
    arr[8] = (0.32, 0.25)
    arr[12] = (0.45, 0.22)
    arr[16] = (0.58, 0.22)
    arr[20] = (0.70, 0.25)

    if kind == "pinch":
        arr[4] = (0.33, 0.27)
        arr[8] = (0.34, 0.28)

    elif kind == "fist":
        for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
            arr[tip] = arr[mcp] + np.array([0.0, 0.04])
        arr[4] = (0.32, 0.50)

    elif kind == "space":
        arr[16] = arr[13] + np.array([0.0, 0.04])
        arr[20] = arr[17] + np.array([0.0, 0.04])
        arr[4] = (0.28, 0.42)

    return arr.tolist()


def run_self_tests() -> pd.DataFrame:
    results: List[Dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append(
            {
                "test": name,
                "passed": bool(condition),
                "detail": detail or ("ok" if condition else "failed"),
            }
        )

    check("distance math", abs(dist((0, 0), (3, 4)) - 5.0) < 1e-9)

    cfg = dict(CONFIG)
    kb = Keyboard(cfg)
    g_center = kb.center("g")
    check("key center exists", g_center is not None)
    check("key mapping center", kb.hit_test(g_center, 0.15) == "g")
    magnetic_point = (g_center[0] + 0.06, g_center[1])
    check("key mapping magnetic", kb.hit_test(magnetic_point, 0.15) == "g")

    cfg_dwell = dict(CONFIG)
    cfg_dwell["dwell_ms"] = 100
    cfg_dwell["refractory_ms"] = 0
    kb_dwell = Keyboard(cfg_dwell)
    sel_dwell = SelectionEngine(cfg_dwell, kb_dwell)
    point = kb_dwell.center("g")
    events: List[GestureEvent] = []
    for t in range(0, 181, 20):
        events.extend(sel_dwell.update(point, t, 0.15, {}, []))
    dwell_types = [e for e in events if e.type == "dwell_type"]
    check("dwell timer emits once", len(dwell_types) == 1, f"count={len(dwell_types)}")

    cfg_pinch = dict(CONFIG)
    cfg_pinch["refractory_ms"] = 0
    kb_pinch = Keyboard(cfg_pinch)
    sel_pinch = SelectionEngine(cfg_pinch, kb_pinch)
    point = kb_pinch.center("f")
    events = []
    events.extend(sel_pinch.update(point, 0, 0.15, {"pinch_ratio": 0.20}, []))
    events.extend(sel_pinch.update(point, 50, 0.15, {"pinch_ratio": 0.20}, []))
    events.extend(sel_pinch.update(point, 100, 0.15, {"pinch_ratio": 0.80}, []))
    check("pinch release types", any(e.type == "pinch_type" for e in events))

    sel_move = SelectionEngine(cfg_pinch, kb_pinch)
    point = kb_pinch.center("f")
    away = (point[0] + 0.25, point[1] - 0.10)
    events = []
    events.extend(sel_move.update(point, 0, 0.15, {"pinch_ratio": 0.20}, []))
    events.extend(sel_move.update(away, 50, 0.15, {"pinch_ratio": 0.20}, []))
    events.extend(sel_move.update(away, 100, 0.15, {"pinch_ratio": 0.80}, []))
    check(
        "pinch move cancels",
        not any(e.type == "pinch_type" for e in events) and any(e.type == "pinch_cancel" for e in events),
    )

    tt = TypingTest("hello")
    for i, ch in enumerate("hello"):
        tt.feed(ch, i * 15000, "test")
    check("WPM calculation", abs(tt.wpm() - 1.0) < 1e-6, f"wpm={tt.wpm():.3f}")
    check("error rate zero", tt.error_rate() == 0.0)

    tt2 = TypingTest("a")
    tt2.feed("b", 0, "test")
    tt2.feed("a", 1000, "test")
    check("error rate calculation", abs(tt2.error_rate() - 0.5) < 1e-9)

    rng = random.Random(1)
    pts = [(0.5 + rng.gauss(0, 0.02), 0.5 + rng.gauss(0, 0.02)) for _ in range(80)]
    filt = PointOneEuroFilter(1.0, 0.1, 1.0)
    filtered_pts = [filt.filter(p, i / 30.0) for i, p in enumerate(pts)]
    raw_std = float(np.std(np.asarray(pts)))
    filt_std = float(np.std(np.asarray(filtered_pts)))
    check("one euro smooths jitter", filt_std < raw_std, f"filtered={filt_std:.4f}, raw={raw_std:.4f}")

    raw_open = classify_raw_gestures(make_synthetic_landmarks("open"), CONFIG)
    check("open palm detected", raw_open.get("open_palm", False))

    raw_fist = classify_raw_gestures(make_synthetic_landmarks("fist"), CONFIG)
    check("fist detected", raw_fist.get("fist", False))

    raw_pinch = classify_raw_gestures(make_synthetic_landmarks("pinch"), CONFIG)
    check("pinch ratio low", raw_pinch.get("pinch_ratio", 1.0) < CONFIG["pinch_on"])

    raw_space = classify_raw_gestures(make_synthetic_landmarks("space"), CONFIG)
    check("space gesture detected", raw_space.get("space", False))

    return pd.DataFrame(results)


BASE_CSS = """
<style>
.ak-privacy {
  background: #064e3b;
  color: #d1fae5;
  padding: 10px 12px;
  border-radius: 10px;
  font-size: 14px;
  line-height: 1.35;
}
.ak-camera {
  background: #7f1d1d;
  color: #fee2e2;
  padding: 10px 12px;
  border-radius: 10px;
  font-weight: 700;
  font-size: 14px;
}
.ak-text {
  min-height: 54px;
  background: #0f172a;
  color: #e2e8f0;
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 12px 14px;
  font-size: 22px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap;
  word-break: break-word;
}
.ak-wrap {
  position: relative;
  width: 100%;
  padding-bottom: 58%;
  background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
  border: 1px solid #1e293b;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
}
.ak-key {
  position: absolute;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(148, 163, 184, .13);
  border: 1px solid rgba(148, 163, 184, .28);
  border-radius: 10px;
  color: #e2e8f0;
  font-weight: 800;
  font-size: clamp(8px, 1.35vw, 16px);
  box-sizing: border-box;
  transition: background .08s linear, transform .08s linear, box-shadow .08s linear;
}
.ak-hover {
  background: rgba(37, 99, 235, .72) !important;
  border-color: #60a5fa !important;
  box-shadow: 0 0 0 2px rgba(96, 165, 250, .35);
}
.ak-pred {
  outline: 2px solid rgba(34, 197, 94, .75);
  outline-offset: -2px;
}
.ak-dot {
  position: absolute;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: #fb923c;
  border: 2px solid #fff7ed;
  transform: translate(-50%, -50%);
  box-shadow: 0 0 14px rgba(251, 146, 60, .85);
  z-index: 5;
}
.ak-ring {
  position: absolute;
  width: 46px;
  height: 46px;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  background: conic-gradient(#22c55e var(--deg, 0deg), rgba(255,255,255,.14) var(--deg, 0deg));
  -webkit-mask: radial-gradient(circle, transparent 54%, #000 55%);
  mask: radial-gradient(circle, transparent 54%, #000 55%);
  z-index: 4;
}
.ak-under {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  margin-top: 8px;
}
.ak-sugg {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.ak-chip {
  background: #111827;
  color: #a7f3d0;
  border: 1px solid #14532d;
  border-radius: 999px;
  padding: 5px 10px;
  font-weight: 700;
  font-size: 14px;
}
.ak-status {
  color: #cbd5e1;
  font-size: 14px;
}
.ak-arm {
  background: #166534;
  color: #bbf7d0;
  border-radius: 999px;
  padding: 3px 8px;
  font-size: 12px;
  font-weight: 800;
  margin-left: 6px;
}
</style>
"""


def privacy_html() -> str:
    return BASE_CSS + (
        "<div class='ak-privacy'>🔒 Local only. No upload. Tell people in view that the camera is on. "
        "log.csv is written only when Record is enabled.</div>"
    )


def camera_notice_html() -> str:
    return BASE_CSS + (
        "<div class='ak-camera'>🎥 CAMERA ON — please tell everyone in view that touchless typing is being demonstrated.</div>"
    )


def text_html(text: str) -> str:
    return BASE_CSS + f"<div class='ak-text'>{esc(text) or '&nbsp;'}</div>"


def keyboard_html(
    kb: Keyboard,
    hover: Optional[str],
    progress: float,
    point: Optional[Tuple[float, float]],
    predictions: Optional[List[str]],
    status: str,
    suggestions: Optional[List[str]],
    armed: bool,
) -> str:
    pred_set = set(predictions or [])
    key_parts: List[str] = []

    for k in kb.keys:
        scale = CONFIG["adaptive_key_scale"] if k.char in pred_set else 1.0
        x = (k.x + k.w * (1.0 - scale) / 2.0) * 100.0
        y = (k.y + k.h * (1.0 - scale) / 2.0) * 100.0
        w = k.w * scale * 100.0
        h = k.h * scale * 100.0

        cls = "ak-key"
        if hover == k.char:
            cls += " ak-hover"
        if k.char in pred_set:
            cls += " ak-pred"

        key_parts.append(
            f'<div class="{cls}" style="left:{x:.2f}%;top:{y:.2f}%;width:{w:.2f}%;height:{h:.2f}%">'
            f"{esc(k.label)}</div>"
        )

    dot_html = ""
    if point is not None:
        deg = int(clamp(progress, 0.0, 1.0) * 360.0)
        left = point[0] * 100.0
        top = point[1] * 100.0
        dot_html = (
            f'<div class="ak-ring" style="left:{left:.2f}%;top:{top:.2f}%;--deg:{deg}deg"></div>'
            f'<div class="ak-dot" style="left:{left:.2f}%;top:{top:.2f}%"></div>'
        )

    sugg_html = "".join(f'<span class="ak-chip">{esc(s)}</span>' for s in (suggestions or []))
    arm_html = "<span class='ak-arm'>ARMED</span>" if armed else ""

    return BASE_CSS + (
        f'<div class="ak-wrap">{"".join(key_parts)}{dot_html}</div>'
        f'<div class="ak-under"><div class="ak-sugg">{sugg_html}</div>'
        f'<div class="ak-status">{esc(status)} {arm_html}</div></div>'
    )


def render_demo_frame(containers: Dict[str, Any], frame: Dict[str, Any], kb: Keyboard) -> None:
    containers["video"].markdown(privacy_html(), unsafe_allow_html=True)
    containers["kb"].markdown(
        keyboard_html(
            kb,
            frame.get("hover"),
            frame.get("progress", 0.0),
            frame.get("point"),
            frame.get("predictions", []),
            frame.get("status", ""),
            frame.get("suggestions", []),
            frame.get("armed", False),
        ),
        unsafe_allow_html=True,
    )
    containers["text"].markdown(text_html(frame.get("text", "")), unsafe_allow_html=True)

    with containers["metrics"].container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Key", display_key(frame.get("hover")))
        c2.metric("Dwell", f"{int(frame.get('progress', 0.0) * 100)}%")
        c3.metric("Armed", "Yes" if frame.get("armed") else "No")
        c4.metric("Clock", f"{frame.get('t_ms', 0) / 1000.0:.1f}s")

    containers["status"].caption(frame.get("status", ""))


def render_live_state(containers: Dict[str, Any], state: Dict[str, Any], kb: Keyboard) -> None:
    if state.get("frame") is not None:
        containers["video"].image(state["frame"], channels="BGR", use_container_width=True)
    else:
        containers["video"].markdown(camera_notice_html(), unsafe_allow_html=True)

    containers["kb"].markdown(
        keyboard_html(
            kb,
            state.get("hover"),
            state.get("progress", 0.0),
            state.get("point"),
            state.get("predictions", []),
            state.get("status", ""),
            state.get("suggestions", []),
            state.get("armed", False),
        ),
        unsafe_allow_html=True,
    )

    containers["text"].markdown(text_html(state.get("text", "")), unsafe_allow_html=True)

    with containers["metrics"].container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("FPS", f"{state.get('fps', 0.0):.1f}")
        c2.metric("Latency", f"{state.get('latency_ms', 0.0):.0f} ms")
        c3.metric("Key", display_key(state.get("hover")))
        c4.metric("Dwell", f"{int(state.get('progress', 0.0) * 100)}%")
        c5.metric("Armed", "Yes" if state.get("armed") else "No")

    containers["status"].caption(state.get("status", ""))


def play_demo(containers: Dict[str, Any]) -> None:
    frames = st.session_state.get("demo_frames", [])
    if not frames:
        return

    kb = st.session_state.keyboard
    speed = float(st.session_state.get("demo_speed", CONFIG["fps"]))
    interval = 1.0 / max(1.0, speed)
    start = time.time()

    for i, frame in enumerate(frames):
        render_demo_frame(containers, frame, kb)
        st.session_state.last_frame = frame
        target_time = i * interval
        delay = target_time - (time.time() - start)
        if delay > 0:
            time.sleep(delay)

    st.session_state.play_demo = False
    st.session_state.text = frames[-1]["text"]


def run_live(
    containers: Dict[str, Any],
    duration_s: int,
    record: bool,
    test_phrase: Optional[str],
) -> None:
    if cv2 is None:
        containers["status"].error("OpenCV is not available. Install requirements and restart Streamlit.")
        return

    pipeline = LivePipeline(
        CONFIG,
        st.session_state.keyboard,
        st.session_state.predictor,
        st.session_state.selection,
        st.session_state.text,
        record,
        test_phrase,
    )

    if not (pipeline.grabber is not None and pipeline.grabber.ok):
        containers["status"].warning("Camera unavailable. You can still use Auto Demo or Simulated Study.")
        st.session_state.detector_backend = pipeline.detector.backend
        pipeline.shutdown()
        return

    st.session_state.detector_backend = pipeline.detector.backend
    pipeline.start()

    end = time.time() + float(duration_s)
    kb = st.session_state.keyboard

    while time.time() < end:
        state = pipeline.get_state()
        render_live_state(containers, state, kb)

        st.session_state.last_frame = {
            "t_ms": state.get("t_ms", 0),
            "point": state.get("point"),
            "hover": state.get("hover"),
            "progress": state.get("progress", 0.0),
            "text": state.get("text", ""),
            "status": state.get("status", ""),
            "suggestions": state.get("suggestions", []),
            "predictions": state.get("predictions", []),
            "armed": state.get("armed", False),
        }

        time.sleep(1.0 / float(CONFIG["fps"]))

    pipeline.shutdown()

    st.session_state.text = pipeline.text
    st.session_state.logs.extend(pipeline.get_logs())

    if pipeline.test is not None:
        st.session_state.live_test_report = {
            "target": pipeline.test.target,
            "wpm": round(pipeline.test.wpm(), 2),
            "error_rate": round(pipeline.test.error_rate(), 4),
            "typed": pipeline.test.typed,
            "correct_prefix": pipeline.test.pos,
        }

    containers["status"].success("Live session finished. In-memory logs are available in Logs & Privacy.")


def render_study(body: Any) -> None:
    df, summary = st.session_state.study
    body.subheader("Simulated 5-user comparison")
    body.caption(
        "This is reproducible demo data for evaluation design. Real evaluation should collect 5 users, "
        "10 phrases per method, and report mean and spread."
    )
    body.dataframe(summary, use_container_width=True)
    body.dataframe(df, use_container_width=True)
    body.download_button(
        "Download simulated study CSV",
        df.to_csv(index=False).encode("utf-8"),
        "airkeys_simulated_study.csv",
        "text/csv",
    )


def delete_all_data(body: Any) -> None:
    try:
        if os.path.exists(CONFIG["log_path"]):
            os.remove(CONFIG["log_path"])
    except Exception as exc:
        body.error(f"Could not delete log.csv: {exc}")

    st.session_state.logs = []
    st.session_state.text = ""
    st.session_state.selection = SelectionEngine(CONFIG, st.session_state.keyboard)
    body.success("Deleted local log.csv and cleared in-session logs/text.")


def render_logs(body: Any) -> None:
    body.subheader("Session log")
    df = pd.DataFrame(st.session_state.logs, columns=["t_ms", "key", "method", "correct"])

    if df.empty:
        body.caption("No user actions recorded yet. Auto-demo logs are shown in Auto Demo and are not written to disk.")
    else:
        body.dataframe(df, use_container_width=True)
        body.download_button(
            "Download log.csv",
            df.to_csv(index=False).encode("utf-8"),
            "log.csv",
            "text/csv",
        )

    body.divider()
    body.subheader("Privacy controls")
    if body.button("Delete all local data"):
        delete_all_data(body)

    body.divider()
    body.subheader("Built-in unit tests")
    if body.button("Run unit tests"):
        st.session_state.test_results = run_self_tests()

    if "test_results" in st.session_state:
        body.dataframe(st.session_state.test_results, use_container_width=True)


def init_session() -> None:
    if "keyboard" not in st.session_state:
        st.session_state.keyboard = Keyboard(CONFIG)

    if "predictor" not in st.session_state:
        st.session_state.predictor = NGramPredictor(CONFIG)

    if "selection" not in st.session_state:
        st.session_state.selection = SelectionEngine(CONFIG, st.session_state.keyboard)

    if "text" not in st.session_state:
        st.session_state.text = ""

    if "logs" not in st.session_state:
        st.session_state.logs = []

    if "demo_frames" not in st.session_state:
        frames, text, logs, report = generate_demo_frames(CONFIG)
        st.session_state.demo_frames = frames
        st.session_state.demo_text = text
        st.session_state.demo_logs = logs
        st.session_state.demo_report = report
        st.session_state.last_frame = frames[-1] if frames else None

    if "study" not in st.session_state:
        st.session_state.study = generate_study(CONFIG)

    if "first_run" not in st.session_state:
        st.session_state.first_run = True
        st.session_state.play_demo = True


def main() -> None:
    st.set_page_config(page_title="AirKeys: Touchless Typing", page_icon="⌨️", layout="wide")
    init_session()

    st.sidebar.title("AirKeys controls")
    st.sidebar.markdown(privacy_html(), unsafe_allow_html=True)

    mode = st.sidebar.radio(
        "Mode",
        ["Auto Demo", "Live Camera", "Simulated Study", "Logs & Privacy"],
    )
    st.session_state.mode = mode

    record = st.sidebar.checkbox(
        "Record user actions to log.csv",
        value=False,
        help="Writes only a local log.csv file when enabled.",
    )
    st.session_state.record = record

    demo_speed = st.sidebar.slider("Auto demo FPS", 5, 30, CONFIG["fps"], key="demo_speed_slider")
    st.session_state.demo_speed = demo_speed

    if st.sidebar.button("Replay Auto Demo"):
        st.session_state.play_demo = True

    if st.sidebar.button("Clear typed text"):
        st.session_state.text = ""
        st.session_state.selection = SelectionEngine(CONFIG, st.session_state.keyboard)

    if mode == "Live Camera":
        duration = st.sidebar.slider("Live session seconds", 5, 30, CONFIG["live_seconds"])
        st.session_state.live_duration = duration

        phrase_choice = st.sidebar.selectbox("Optional typing-test phrase", ["— none —"] + TEST_PHRASES)
        st.session_state.test_phrase = None if phrase_choice == "— none —" else phrase_choice

        if st.sidebar.button("Start live session"):
            st.session_state.run_live = True

        st.sidebar.info(
            "Camera processing is local. If a MediaPipe Tasks hand model file exists, it is used. "
            "Otherwise the app falls back to local MediaPipe legacy tracking when available. "
            "If no camera is available, use Auto Demo."
        )

    st.title("AirKeys: Touchless Typing")
    st.caption(
        "Touchless QWERTY with dwell selection, pinch-with-release selection, gesture editing, "
        "word suggestions, and a typing test. All processing stays on this computer."
    )

    containers = {
        "video": st.empty(),
        "kb": st.empty(),
        "text": st.empty(),
        "metrics": st.empty(),
        "status": st.empty(),
        "body": st.empty(),
    }

    if st.session_state.get("play_demo"):
        play_demo(containers)

    if st.session_state.get("run_live"):
        st.session_state.run_live = False
        run_live(
            containers,
            int(st.session_state.get("live_duration", CONFIG["live_seconds"])),
            bool(st.session_state.get("record", False)),
            st.session_state.get("test_phrase"),
        )

    if mode == "Auto Demo":
        frame = st.session_state.get("last_frame")
        if frame:
            render_demo_frame(containers, frame, st.session_state.keyboard)

        with containers["body"].container():
            st.subheader("Automatic demo report")
            rep = st.session_state.demo_report
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Characters", rep["characters"])
            c2.metric("WPM", rep["wpm"])
            c3.metric("Error rate", f"{rep['error_rate'] * 100:.1f}%")
            c4.metric("Duration", f"{rep['duration_s']}s")

            st.dataframe(
                pd.DataFrame(st.session_state.demo_logs, columns=["t_ms", "key", "method", "correct"]),
                use_container_width=True,
            )

    elif mode == "Live Camera":
        frame = st.session_state.get("last_frame")
        if frame:
            render_demo_frame(containers, frame, st.session_state.keyboard)
        else:
            containers["video"].markdown(camera_notice_html(), unsafe_allow_html=True)
            containers["kb"].markdown(
                keyboard_html(
                    st.session_state.keyboard,
                    None,
                    0.0,
                    None,
                    [],
                    "Press Start live session, or switch to Auto Demo for an immediate simulation.",
                    [],
                    False,
                ),
                unsafe_allow_html=True,
            )
            containers["text"].markdown(text_html(st.session_state.text), unsafe_allow_html=True)

        with containers["body"].container():
            st.subheader("Live camera notes")
            st.markdown(
                "- Dwell: hold the index fingertip over a key.\n"
                "- Pinch: start pinch over a key, keep still, release to type.\n"
                "- Space: two-finger gesture.\n"
                "- Backspace: open palm for 1 second to arm, then fist.\n"
                f"- Detector backend: `{st.session_state.get('detector_backend', 'not started yet')}`"
            )

            if "live_test_report" in st.session_state:
                rep = st.session_state.live_test_report
                st.subheader("Live typing-test report")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Target", rep["target"])
                c2.metric("WPM", rep["wpm"])
                c3.metric("Error rate", f"{rep['error_rate'] * 100:.1f}%")
                c4.metric("Correct prefix", f"{rep['correct_prefix']}/{len(rep['target'])}")

    elif mode == "Simulated Study":
        containers["video"].markdown(privacy_html(), unsafe_allow_html=True)
        render_study(containers["body"])

    else:
        containers["video"].markdown(privacy_html(), unsafe_allow_html=True)
        render_logs(containers["body"])


main()