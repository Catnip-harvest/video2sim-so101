"""Out-of-distribution (OOD) ladder: edit a Variation AFTER sample_variation (docs/OOD_LADDER.md).

  v = sample_variation(spec, seed, idx, variety=True)
  v = apply_ood(v, "camera")          # "none" returns v itself, untouched

Each rung draws from its own stream, numpy.random.default_rng([v.seed, TAG]), so the base
draws (physics, cube pose, expert tempo) are never consumed or changed, and "combo" uses
exactly the camera draws and exactly the colour draws of the same episode. What was changed
is written to v.extras["ood"] (it lands in episodes.jsonl with the rest of the Variation).

Image-level exposure is not a scene property: it is stored in v.extras["ood"]["exposure"]
and applied by eval/sim_eval.py's ObsRenderer to every rendered frame (both cameras), before
the h264 round-trip, via apply_exposure().
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

_DG = str(Path(__file__).resolve().parents[1] / "datagen")
if _DG not in sys.path:
  sys.path.insert(0, _DG)

from r2s.variations import (FRONT_CAM_LOOKAT, FRONT_CAM_POS, Variation, _hsv,  # noqa: E402
                            _pitch_rotate)

RUNGS = ("none", "camera", "camera_height", "camera_pitch", "camera_yaw", "camera_distance",
         "exposure", "colour", "combo", "distractor")

# One stream tag per factor (fixed forever: changing one changes that rung's scenes).
TAG_CAMERA, TAG_EXPOSURE, TAG_COLOUR, TAG_DISTRACTOR = 101, 102, 103, 104

FOVY_NOMINAL = 45.0

# Training bands (variations.py) and the OOD magnitudes (|value| drawn uniformly, random sign).
CAM_HEIGHT_OOD = (0.12, 0.16)     # m      training +-0.08
CAM_PITCH_OOD = (12.0, 15.0)      # deg    training +-8
CAM_YAW_OOD = (10.0, 15.0)        # deg    training: implicit ~+-3.4 (x/y +-3 cm at 0.53 m)
CAM_DIST_OOD = (0.08, 0.12)       # m      training: implicit ~+-5 cm
CAM_FOVY_OOD = FOVY_NOMINAL + 6.0  # deg   training 45 +-3

EXPOSURE_CHOICES = (
  ("brightness", 0.5), ("brightness", 1.6),
  ("contrast", 0.6), ("contrast", 1.4),
  ("white_balance", None),  # R and B gains each drawn from {0.8, 1.2}, G = 1
)
WB_GAINS = (0.8, 1.2)

# Colour rung (training: cube s,v 0.5-0.95; bin s,v 0.3-0.9; table s 0-0.35, v 0.35-0.9).
CUBE_MAGENTA = _hsv(0.83, 1.0, 1.0)
BIN_BLACK = [0.05, 0.05, 0.05]
TABLE_GREEN = (_hsv(0.33, 0.85, 0.55), _hsv(0.33, 0.85, 0.45))
# Lights: 0.5x the training minima or 1.6x the training maxima.
LIGHT_DIM = {"light_diffuse": 0.5 * 0.35, "ambient": 0.5 * 0.08, "headlight": 0.5 * 0.15}
LIGHT_BRIGHT = {"light_diffuse": 1.6 * 0.9, "ambient": 1.6 * 0.35, "headlight": 1.6 * 0.5}


# ---- camera ---------------------------------------------------------------------------------
def front_camera(dxy, dz: float, dpitch_deg: float, dyaw_deg: float = 0.0,
                 ddist: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
  """Front camera pose from offsets around the nominal pose. With dyaw = ddist = 0 this is
  exactly variations.sample_variation's construction (translate pos and aim point by dxy,
  raise pos by dz, re-aim at the translated nominal point, tilt by dpitch). Yaw orbits the
  camera about the vertical axis through the aim point; distance moves it along the view
  axis (+ = farther). Pitch is applied last, about the camera's own horizontal axis."""
  dxy = np.asarray(dxy, float)
  pos = FRONT_CAM_POS + np.array([dxy[0], dxy[1], dz])
  aim = FRONT_CAM_LOOKAT + np.array([dxy[0], dxy[1], 0.0])
  if dyaw_deg:
    a = math.radians(dyaw_deg)
    r = pos[:2] - aim[:2]
    pos = pos.copy()
    pos[:2] = aim[:2] + np.array([math.cos(a) * r[0] - math.sin(a) * r[1],
                                  math.sin(a) * r[0] + math.cos(a) * r[1]])
  if ddist:
    f = (pos - aim) / np.linalg.norm(pos - aim)
    pos = pos + f * ddist
  lookat = _pitch_rotate(pos, aim, math.radians(dpitch_deg))
  return pos, lookat


def _signed(rng: np.random.Generator, lo: float, hi: float) -> float:
  mag = float(rng.uniform(lo, hi))
  return mag if rng.uniform() < 0.5 else -mag


def _camera(v: Variation, which: str) -> dict:
  """Edit v's front camera in place; returns the record of what was set."""
  rng = np.random.default_rng([v.seed, TAG_CAMERA])
  # Draw every axis in a fixed order whatever the rung, so a given id gets the same
  # magnitude for an axis in "camera" and in its single-axis sub-rung.
  h, p, y, d = (_signed(rng, *CAM_HEIGHT_OOD), _signed(rng, *CAM_PITCH_OOD),
                _signed(rng, *CAM_YAW_OOD), _signed(rng, *CAM_DIST_OOD))
  dxy = (np.array(v.cam_pos) - FRONT_CAM_POS)[:2]  # keep the episode's in-range x/y draw
  dz, dpitch, dyaw, ddist, fovy = v.cam_dz, v.cam_dpitch_deg, 0.0, 0.0, v.cam_fovy
  if which == "camera":
    dz, dpitch, fovy = h, p, CAM_FOVY_OOD
  elif which == "camera_height":
    dz = h
  elif which == "camera_pitch":
    dpitch = p
  elif which == "camera_yaw":
    dyaw = y
  elif which == "camera_distance":
    ddist = d
  else:
    raise ValueError(which)
  pos, lookat = front_camera(dxy, dz, dpitch, dyaw, ddist)
  v.cam_pos = [float(x) for x in pos]
  v.cam_lookat = [float(x) for x in lookat]
  v.cam_fovy = float(fovy)
  v.cam_dz = float(dz)
  v.cam_dpitch_deg = float(dpitch)
  dist = float(np.linalg.norm(pos - (FRONT_CAM_LOOKAT + np.array([dxy[0], dxy[1], 0.0]))))
  return {"rung": which, "dz_m": round(dz, 4), "dpitch_deg": round(dpitch, 3),
          "dyaw_deg": round(dyaw, 3), "ddist_m": round(ddist, 4), "fovy_deg": round(fovy, 3),
          "dxy_m": [round(float(x), 4) for x in dxy], "dist_to_aim_m": round(dist, 4),
          "perturbed": {"camera": ["height", "pitch", "fov"], "camera_height": ["height"],
                        "camera_pitch": ["pitch"], "camera_yaw": ["yaw"],
                        "camera_distance": ["distance"]}[which]}


# ---- exposure -------------------------------------------------------------------------------
def _exposure(v: Variation) -> dict:
  rng = np.random.default_rng([v.seed, TAG_EXPOSURE])
  kind, level = EXPOSURE_CHOICES[int(rng.integers(len(EXPOSURE_CHOICES)))]
  rec = {"kind": kind, "brightness": 1.0, "contrast": 1.0, "wb_gain": [1.0, 1.0, 1.0]}
  if kind == "white_balance":
    rec["wb_gain"] = [float(WB_GAINS[int(rng.integers(2))]), 1.0, float(WB_GAINS[int(rng.integers(2))])]
  else:
    rec[kind] = float(level)
  return rec


def apply_exposure(img: np.ndarray, brightness: float = 1.0, contrast: float = 1.0,
                   wb_gain=(1.0, 1.0, 1.0), gamma: float = 1.0, noise_std: float = 0.0,
                   blur_sigma: float = 0.0, rng: np.random.Generator | None = None,
                   **_ignored) -> np.ndarray:
  """Camera-style image transform on an HxWx3 uint8 frame, in this order: white-balance
  gains per channel, brightness gain, contrast about mid-grey (0.5), gamma (out = in**(1/g)),
  Gaussian blur, additive Gaussian noise (std in [0,1] units); clip; round to uint8.
  The identity settings return the frame unchanged (bit for bit)."""
  ident = (brightness == 1.0 and contrast == 1.0 and tuple(wb_gain) == (1.0, 1.0, 1.0)
           and gamma == 1.0 and noise_std == 0.0 and blur_sigma == 0.0)
  if ident:
    return img
  x = img.astype(np.float32) / 255.0
  x = x * np.asarray(wb_gain, np.float32)[None, None, :] * np.float32(brightness)
  x = (x - 0.5) * np.float32(contrast) + 0.5
  x = np.clip(x, 0.0, 1.0)
  if gamma != 1.0:
    x = x ** np.float32(1.0 / gamma)
  if blur_sigma > 0:
    from scipy.ndimage import gaussian_filter
    x = gaussian_filter(x, sigma=(blur_sigma, blur_sigma, 0))
  if noise_std > 0:
    rng = rng if rng is not None else np.random.default_rng(0)
    x = x + rng.normal(0.0, noise_std, x.shape).astype(np.float32)
  return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def exposure_of(v: Variation) -> dict | None:
  return ((v.extras or {}).get("ood") or {}).get("exposure")


# ---- colour ---------------------------------------------------------------------------------
def _colour(v: Variation) -> dict:
  rng = np.random.default_rng([v.seed, TAG_COLOUR])
  bright = bool(rng.uniform() < 0.5)
  v.object_rgb = list(CUBE_MAGENTA)
  v.bin_rgb = list(BIN_BLACK)
  v.table_rgb1, v.table_rgb2 = list(TABLE_GREEN[0]), list(TABLE_GREEN[1])
  lights = LIGHT_BRIGHT if bright else LIGHT_DIM
  v.light_diffuse = float(lights["light_diffuse"])
  v.ambient = float(lights["ambient"])
  v.headlight = float(lights["headlight"])
  return {"cube_rgb": v.object_rgb, "bin_rgb": v.bin_rgb, "table_rgb": [v.table_rgb1, v.table_rgb2],
          "lights": "bright_1.6x_max" if bright else "dim_0.5x_min", **{k: round(x, 4) for k, x in lights.items()}}


# ---- entry point ----------------------------------------------------------------------------
def apply_ood(v: Variation, rung: str) -> Variation:
  """Return the Variation for this rung. "none" returns ``v`` itself (no copy, no edit)."""
  if rung == "none":
    return v
  if rung not in RUNGS:
    raise ValueError(f"unknown --ood rung {rung!r}; choose from {RUNGS}")
  v = copy.deepcopy(v)
  rec: dict = {"rung": rung}
  if rung.startswith("camera"):
    rec["camera"] = _camera(v, rung)
  elif rung == "exposure":
    rec["exposure"] = _exposure(v)
  elif rung == "colour":
    rec["colour"] = _colour(v)
  elif rung == "combo":
    rec["camera"] = _camera(v, "camera")
    rec["colour"] = _colour(v)
  elif rung == "distractor":
    from r2s.free_bodies import sample_free_bodies
    rng = np.random.default_rng([v.seed, TAG_DISTRACTOR])
    bodies = sample_free_bodies(v, rng, int(rng.integers(1, 3)), shapes=("cylinder", "sphere"),
                                size_range=(0.020, 0.035))
    v.extras = {**(v.extras or {}), "free_bodies": bodies}
    rec["distractor"] = {"n": len(bodies), "shapes": [b["shape"] for b in bodies]}
  v.extras = {**(v.extras or {}), "ood": rec}
  return v
