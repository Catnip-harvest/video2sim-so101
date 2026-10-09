"""MediaPipe Hands (palm detector + 21-keypoint landmark model) run as ONNX
through OpenCV's DNN module.

Why not the `mediapipe` pip package: on this laptop Windows Application Control
blocks its unsigned native library (libmediapipe.dll, "An Application Control
policy has blocked this file"). The OpenCV Zoo publishes the same MediaPipe hand
models converted TFLite -> ONNX (Apache-2.0, opencv/palm_detection_mediapipe and
opencv/handpose_estimation_mediapipe on Hugging Face). Pre/post-processing below
follows the OpenCV Zoo reference (mp_palmdet.py / mp_handpose.py, Apache-2.0),
re-implemented here, plus MediaPipe-style tracking: once a hand is found, the
next frame's crop comes from the previous landmarks instead of re-detecting.

Outputs per hand: 21 image keypoints (px), 21 world keypoints (metres, hand-
centred, average-hand size), handedness, confidence - same as MediaPipe Tasks.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PALM_INPUT = 192
HAND_INPUT = 224
PALM_IDS = [0, 5, 9, 13, 17, 1, 2]   # 7 palm keypoints, in the palm detector's order


def _anchors() -> np.ndarray:
    out = []
    for stride, per_cell in ((8, 2), (16, 6)):
        grid = PALM_INPUT // stride
        for y in range(grid):
            for x in range(grid):
                out += [((x + 0.5) / grid, (y + 0.5) / grid)] * per_cell
    return np.array(out, np.float32)


class PalmDetector:
    def __init__(self, model_path: Path, score_threshold: float = 0.5, nms_threshold: float = 0.3) -> None:
        self.net = cv2.dnn.readNet(str(model_path))
        self.anchors = _anchors()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        """Returns (N, 19): box x1,y1,x2,y2 | 7 palm keypoints (x,y) | score."""
        h, w = image_bgr.shape[:2]
        ratio = PALM_INPUT / max(h, w)
        rh, rw = int(h * ratio), int(w * ratio)
        img = cv2.resize(image_bgr, (rw, rh))
        left, top = (PALM_INPUT - rw) // 2, (PALM_INPUT - rh) // 2
        img = cv2.copyMakeBorder(img, top, PALM_INPUT - rh - top, left, PALM_INPUT - rw - left,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        reg = next(o for o in outs if o.shape[-1] == 18)[0]
        score = 1.0 / (1.0 + np.exp(-next(o for o in outs if o.shape[-1] == 1)[0, :, 0].astype(np.float64)))
        scale = max(h, w)
        pad = np.array([left, top]) / ratio
        cxy = reg[:, :2] / PALM_INPUT + self.anchors
        wh = reg[:, 2:4] / PALM_INPUT
        boxes = np.concatenate([(cxy - wh / 2) * scale - pad, (cxy + wh / 2) * scale - pad], 1)
        keep = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
                                score.tolist(), self.score_threshold, self.nms_threshold)
        keep = np.array(keep).reshape(-1)
        if len(keep) == 0:
            return np.empty((0, 19))
        lms = reg[keep, 4:].reshape(-1, 7, 2) / PALM_INPUT + self.anchors[keep][:, None, :]
        lms = lms * scale - pad
        return np.column_stack([boxes[keep], lms.reshape(-1, 14), score[keep]])


class HandLandmarks:
    def __init__(self, model_path: Path, conf_threshold: float = 0.5) -> None:
        self.net = cv2.dnn.readNet(str(model_path))
        self.conf_threshold = conf_threshold

    @staticmethod
    def _crop_square(image, bbox, shift, enlarge, for_rotation):
        wh = bbox[1] - bbox[0]
        bbox = bbox + np.array(shift) * wh
        center = bbox.sum(0) / 2
        half = (bbox[1] - bbox[0]) * enlarge / 2
        bbox = np.array([center - half, center + half]).astype(np.int32)
        bbox[:, 0] = np.clip(bbox[:, 0], 0, image.shape[1])
        bbox[:, 1] = np.clip(bbox[:, 1], 0, image.shape[0])
        crop = image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        side = int(np.linalg.norm(crop.shape[:2])) if for_rotation else int(max(crop.shape[:2]))
        pad_h, pad_w = side - crop.shape[0], side - crop.shape[1]
        left, top = pad_w // 2, pad_h // 2
        crop = cv2.copyMakeBorder(crop, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return crop, bbox, bbox[0] - np.array([left, top])

    def infer(self, image_bgr: np.ndarray, palm: np.ndarray):
        """palm: (18,) box + 7 palm keypoints. Returns dict or None."""
        palm_box = palm[:4].reshape(2, 2)
        crop, _, bias = self._crop_square(image_bgr, palm_box, (0, 0), 4.0, True)
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            return None
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        lm7 = palm[4:18].reshape(7, 2) - bias
        p1, p2 = lm7[0], lm7[2]       # wrist (palm base) and middle-finger base
        radians = np.pi / 2 - np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0])
        radians -= 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = np.rad2deg(radians)
        rbox_in = palm_box - bias
        center = rbox_in.sum(0) / 2
        rot = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle), 1.0)
        rotated = cv2.warpAffine(crop, rot, (crop.shape[1], crop.shape[0]))
        rlm = np.c_[lm7, np.ones(7)] @ rot.T
        rbox = np.array([rlm.min(0), rlm.max(0)])
        hand_crop, rbox, _ = self._crop_square(rotated, rbox, (0, -0.4), 3.0, False)
        if hand_crop.size == 0:
            return None
        blob = cv2.resize(hand_crop, (HAND_INPUT, HAND_INPUT), interpolation=cv2.INTER_AREA).astype(np.float32)[None] / 255.0
        self.net.setInput(blob)
        names = self.net.getUnconnectedOutLayersNames()
        outs = dict(zip(names, self.net.forward(names)))
        landmarks = outs["Identity"][0].reshape(21, 3).astype(np.float64)
        conf = float(outs["Identity_1"][0, 0])
        handed = float(outs["Identity_2"][0, 0])
        world = outs["Identity_3"][0].reshape(21, 3).astype(np.float64)
        if conf < self.conf_threshold:
            return None
        scale = max((rbox[1] - rbox[0]) / HAND_INPUT)
        landmarks[:, :2] = (landmarks[:, :2] - HAND_INPUT / 2) * scale
        landmarks[:, 2] *= scale
        back = cv2.getRotationMatrix2D((0, 0), float(angle), 1.0)[:, :2]
        lm_xy = landmarks[:, :2] @ back
        world_xy = world[:, :2] @ back
        inv_rot = np.array([[rot[0, 0], rot[1, 0]], [rot[0, 1], rot[1, 1]]])
        inv_t = -inv_rot @ rot[:, 2]
        box_center = np.append(rbox.sum(0) / 2, 1.0)
        original_center = np.array([box_center @ np.r_[inv_rot[0], inv_t[0]], box_center @ np.r_[inv_rot[1], inv_t[1]]])
        image_xy = lm_xy + original_center + bias
        return {"image_px": image_xy, "world_m": np.c_[world_xy, world[:, 2]], "conf": conf,
                "handedness": "Right" if handed > 0.5 else "Left"}


def palm_from_landmarks(image_px: np.ndarray) -> np.ndarray:
    """Tracking: synthesise a palm detection (box + 7 keypoints) from last frame's 21 keypoints."""
    pts = image_px[PALM_IDS]
    box = np.r_[pts.min(0), pts.max(0)]
    return np.r_[box, pts.reshape(-1)]


class OnnxHandTracker:
    def __init__(self, palm_model: Path, hand_model: Path, conf_threshold: float = 0.5) -> None:
        self.palm = PalmDetector(palm_model)
        self.hand = HandLandmarks(hand_model, conf_threshold)
        self.previous = None

    def __call__(self, frame_bgr: np.ndarray):
        result = None
        if self.previous is not None:
            result = self.hand.infer(frame_bgr, palm_from_landmarks(self.previous))
        if result is None:
            candidates = []
            for palm in self.palm.detect(frame_bgr):
                r = self.hand.infer(frame_bgr, palm[:18])
                if r is not None:
                    jump = 0.0 if self.previous is None else float(np.linalg.norm(r["image_px"][0] - self.previous[0]))
                    candidates.append((jump - 1000 * r["conf"], r))
            if candidates:
                result = min(candidates, key=lambda c: c[0])[1]
        self.previous = None if result is None else result["image_px"]
        return result
