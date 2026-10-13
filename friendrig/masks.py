"""Real-image masks for the friend's rig (cam_head, 640x480 RGB uint8).

Arm: the arm is black on a beige table. A pixel is "dark" if max(R,G,B) < DARK_V and it is not
green (shadowed cube faces are dark too). Closing fills specular speckle on the black plastic.
The wrist camera's black USB cable is also dark and is ~6-14 px wide. Instead of pretending it is
arm or table, we mark it IGNORE: dark pixels that a disk opening of CABLE_PX removes are thin
structures (cable, black wire in the servo leads, far-away jaw tips) and are left out of the IoU
on BOTH sides. The raw IoU (no ignore band) is always reported next to it.
"""

from __future__ import annotations

import cv2
import numpy as np

DARK_V = 75
SHEEN_V = 130      # lit faces of the black plastic are blue-grey: V < SHEEN_V and B > R + SHEEN_DB
SHEEN_DB = 8       # (the beige table and its shadows have R >= B; measured ep140 f238)
CLOSE_PX = 5
CABLE_PX = 15


def disk(px: int):
  return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))


def scale_px(px: int, width: int) -> int:
  return max(1, int(round(px * width / 640)))


def dark_mask(img: np.ndarray, dark_v: int = DARK_V) -> np.ndarray:
  x = img.astype(np.int16)
  r, g, b = x[..., 0], x[..., 1], x[..., 2]
  v = x.max(-1)
  dark = (v < dark_v) | ((v < SHEEN_V) & (b > r + SHEEN_DB))
  green = (g > r + 12) & (g > b + 4)
  m = (dark & ~green).astype(np.uint8)
  return cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(CLOSE_PX)).astype(bool)


def arm_masks(img: np.ndarray, dark_v: int = DARK_V, cable_px: int = CABLE_PX):
  """-> (arm, ignore): arm = thick dark structures; ignore = thin dark structures (cable...)."""
  d = dark_mask(img, dark_v)
  core = cv2.morphologyEx(d.astype(np.uint8), cv2.MORPH_OPEN, disk(cable_px)).astype(bool)
  # grow the core back a little so its own boundary pixels are not called "thin"
  ignore = d & ~core
  return core, ignore


def resize_bool(m: np.ndarray, w: int, h: int) -> np.ndarray:
  if m.shape[1] == w and m.shape[0] == h:
    return m.astype(bool)
  return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5


def iou(a: np.ndarray, b: np.ndarray, ignore: np.ndarray | None = None) -> float:
  if ignore is not None:
    keep = ~ignore
    a, b = a & keep, b & keep
  u = np.logical_or(a, b).sum()
  return float(np.logical_and(a, b).sum() / u) if u else 1.0


def green_mask(img: np.ndarray) -> np.ndarray:
  """Green cube: hue in the green band, saturated enough, not too dark."""
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
  m = (h >= 55) & (h <= 95) & (s >= 90) & (v >= 40)
  m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(5))
  return m.astype(bool)


def white_mask(img: np.ndarray) -> np.ndarray:
  """White bowl: bright and unsaturated (table glare is also bright: callers keep the largest blob)."""
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
  m = (s <= 30) & (v >= 200)
  m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(5))
  return m.astype(bool)
