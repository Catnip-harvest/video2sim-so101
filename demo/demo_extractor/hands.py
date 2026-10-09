"""Hand keypoints per frame (MediaPipe HandLandmarker, Apache-2.0) and their
lift into the camera frame.

MediaPipe gives, per hand, 21 image landmarks and 21 "world" landmarks in
metres around the hand's centre, sized for an average adult hand. Knowing the
camera intrinsics, the hand's 3D translation is the one that makes the world
landmarks project onto the image landmarks: a 42-equation linear least-squares
problem. The only unknown left is the true hand size (people differ by ~+-15%),
which shows up as one global scale factor about the camera centre; the
pipeline calibrates it later from the grasp contact (see geometry.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP = 0, 4, 5, 8, 9, 12, 17


@dataclass
class HandTrack:
    image_px: np.ndarray       # (T, 21, 2) processing-resolution pixels, NaN where missing
    world_m: np.ndarray        # (T, 21, 3) MediaPipe world landmarks, NaN where missing
    score: np.ndarray          # (T,) handedness score, 0 where missing
    handedness: list[str]      # per frame, "" where missing

    @property
    def detected(self) -> np.ndarray:
        return ~np.isnan(self.image_px[:, 0, 0])


def detect_hands(frames: list[np.ndarray], times: np.ndarray, model_dir: Path,
                 min_confidence: float = 0.3, backend: str = "auto") -> tuple[HandTrack, str]:
    """backend: "mediapipe" (pip package), "onnx" (same models via OpenCV DNN),
    or "auto" (mediapipe if its native library loads, else onnx)."""
    if backend in ("auto", "mediapipe"):
        try:
            return _detect_mediapipe(frames, times, model_dir / "mediapipe_hand_landmarker" / "hand_landmarker.task",
                                     min_confidence), "mediapipe_tasks"
        except OSError as exc:   # e.g. Windows Application Control blocking libmediapipe.dll
            if backend == "mediapipe":
                raise
            print(f"mediapipe native library unavailable ({exc}); using the ONNX/OpenCV backend", flush=True)
    return _detect_onnx(frames, model_dir, max(min_confidence, 0.5)), "mediapipe_models_onnx_opencv_dnn"


def _detect_onnx(frames: list[np.ndarray], model_dir: Path, conf: float) -> HandTrack:
    from .hand_onnx import OnnxHandTracker

    tracker = OnnxHandTracker(model_dir / "opencv_zoo_palmdet" / "palm_detection_mediapipe_2023feb.onnx",
                              model_dir / "opencv_zoo_handpose" / "handpose_estimation_mediapipe_2023feb.onnx", conf)
    n = len(frames)
    image_px = np.full((n, 21, 2), np.nan)
    world_m = np.full((n, 21, 3), np.nan)
    score = np.zeros(n)
    handedness = [""] * n
    for k, frame in enumerate(frames):
        r = tracker(frame)
        if r is None:
            continue
        image_px[k] = r["image_px"]
        world_m[k] = r["world_m"]
        score[k] = r["conf"]
        handedness[k] = r["handedness"]
    return HandTrack(image_px=image_px, world_m=world_m, score=score, handedness=handedness)


def _detect_mediapipe(frames: list[np.ndarray], times: np.ndarray, model_path: Path,
                      min_confidence: float) -> HandTrack:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    n = len(frames)
    image_px = np.full((n, 21, 2), np.nan)
    world_m = np.full((n, 21, 3), np.nan)
    score = np.zeros(n)
    handedness = [""] * n
    previous_wrist = None
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for k, frame in enumerate(frames):
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                 int(round(times[k] * 1000)))
            if not result.hand_landmarks:
                continue
            # one demonstrating hand: keep continuity with the previous frame, else the most confident
            candidates = []
            for i, landmarks in enumerate(result.hand_landmarks):
                pts = np.array([[p.x * w, p.y * h] for p in landmarks])
                conf = result.handedness[i][0].score if result.handedness else 0.5
                jump = 0.0 if previous_wrist is None else np.linalg.norm(pts[0] - previous_wrist) / max(w, h)
                candidates.append((jump - 0.2 * conf, i, pts, conf))
            _, i, pts, conf = min(candidates, key=lambda c: c[0])
            image_px[k] = pts
            world_m[k] = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]])
            score[k] = conf
            handedness[k] = result.handedness[i][0].category_name if result.handedness else ""
            previous_wrist = pts[0]
    return HandTrack(image_px=image_px, world_m=world_m, score=score, handedness=handedness)


def lift_to_camera(track: HandTrack, K: np.ndarray, z_sign: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame landmarks (T,21,3) at average-hand scale, and per-frame RMS
    reprojection error in px. z_sign flips MediaPipe's world z if needed."""
    n = len(track.score)
    out = np.full((n, 21, 3), np.nan)
    rms = np.full(n, np.nan)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    for k in np.where(track.detected)[0]:
        world = track.world_m[k] * np.array([1.0, 1.0, z_sign])
        xn = (track.image_px[k, :, 0] - cx) / fx
        yn = (track.image_px[k, :, 1] - cy) / fy
        # x_n (Z_j + Tz) = X_j + Tx  ->  Tx - x_n Tz = x_n Z_j - X_j   (same for y)
        A = np.zeros((42, 3))
        b = np.zeros(42)
        A[:21, 0] = 1.0
        A[:21, 2] = -xn
        b[:21] = xn * world[:, 2] - world[:, 0]
        A[21:, 1] = 1.0
        A[21:, 2] = -yn
        b[21:] = yn * world[:, 2] - world[:, 1]
        translation, *_ = np.linalg.lstsq(A, b, rcond=None)
        pts = world + translation
        if np.any(pts[:, 2] <= 0.05):
            continue
        proj = np.stack([fx * pts[:, 0] / pts[:, 2] + cx, fy * pts[:, 1] / pts[:, 2] + cy], -1)
        out[k] = pts
        rms[k] = float(np.sqrt(np.mean(np.sum((proj - track.image_px[k]) ** 2, -1))))
    return out, rms


def choose_world_z_sign(track: HandTrack, K: np.ndarray) -> float:
    """MediaPipe's world-landmark z convention is checked against the image:
    the sign that reprojects better wins (decided once per video)."""
    errors = {}
    for sign in (1.0, -1.0):
        _, rms = lift_to_camera(track, K, sign)
        errors[sign] = np.nanmedian(rms) if np.any(np.isfinite(rms)) else np.inf
    return min(errors, key=errors.get)


def fill_and_smooth(points: np.ndarray, max_gap: int, window: int = 5) -> np.ndarray:
    """Linear-interpolate short gaps (<= max_gap frames), then Savitzky-Golay
    smooth each contiguous run. points: (T, J, 3) with NaN rows for misses."""
    out = points.copy()
    valid = ~np.isnan(out[:, 0, 0])
    idx = np.arange(len(out))
    if valid.sum() < 2:
        return out
    # fill short gaps
    gaps = np.where(~valid)[0]
    for g in gaps:
        prev = idx[valid & (idx < g)]
        nxt = idx[valid & (idx > g)]
        if len(prev) and len(nxt) and (nxt[0] - prev[-1] - 1) <= max_gap:
            u = (g - prev[-1]) / (nxt[0] - prev[-1])
            out[g] = (1 - u) * points[prev[-1]] + u * points[nxt[0]]
    valid = ~np.isnan(out[:, 0, 0])
    # smooth contiguous runs
    run_start = None
    for k in range(len(out) + 1):
        if k < len(out) and valid[k]:
            run_start = k if run_start is None else run_start
            continue
        if run_start is not None:
            run = slice(run_start, k)
            length = k - run_start
            win = min(window, length if length % 2 == 1 else length - 1)
            if win >= 3:
                out[run] = savgol_filter(out[run], win, 2, axis=0)
            run_start = None
    return out
