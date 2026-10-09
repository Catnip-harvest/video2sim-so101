"""--wide: wider per-episode front-camera pose and image exposure (P2.5 data-engine experiment).

What changes with --wide, per episode (the camera stays FIXED for the whole episode):
  front camera   height +-15 cm, pitch +-15 deg, yaw +-15 deg (orbit about the vertical axis
                 through the aim point), distance +-10 cm along the view axis, FOV 45 +-8 deg,
                 all around the same nominal pose as the normal draw (and keeping that draw's
                 own x/y +-3 cm offset). Same construction as eval/ood.py's front_camera().
  exposure       applied to the RENDERED frames of BOTH cameras, before h264 encoding:
                 brightness x[0.75, 1.3], contrast x[0.8, 1.2] about mid-grey, white-balance
                 gain per channel [0.9, 1.1], gamma [0.85, 1.15], Gaussian noise std
                 [0, 3/255], Gaussian blur sigma [0.3, 0.7] px in 30 % of episodes.
  The wrist camera pose is unchanged.

Every --wide draw comes from its own streams, numpy.random.default_rng([episode_seed, TAG]),
so every other draw (layout, physics, servos, look, expert variety) is identical to the
non-wide run with the same seed/attempt: same scenes, same expert trajectories, same
accept/reject verdicts. Only the front camera and the image exposure differ.

Visibility: generate.py renders the first recorded frame's front-camera segmentation and
redraws the camera (next draw from the same camera stream, up to MAX_CAMERA_DRAWS) while the
cube or the gripper covers fewer than MIN_VISIBLE_PX pixels (at 256x256, scaled by area).
If no draw passes, the episode keeps its normal (narrow) camera and is flagged. This never
touches the physics: it runs after the verdict and only changes camera parameters.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .variations import FRONT_CAM_FOVY, FRONT_CAM_LOOKAT, FRONT_CAM_POS, Variation, _pitch_rotate

# Stream tags (fixed forever: changing one changes the wide dataset).
TAG_WIDE_CAMERA, TAG_WIDE_EXPOSURE, TAG_WIDE_NOISE = 211, 212, 213

# Half-widths of the uniform draws around the nominal pose.
WIDE_RANGES = {"dz_m": 0.15, "dpitch_deg": 15.0, "dyaw_deg": 15.0, "ddist_m": 0.10,
               "dfovy_deg": 8.0}
EXPOSURE_RANGES = {"brightness": (0.75, 1.3), "contrast": (0.8, 1.2), "wb_gain": (0.9, 1.1),
                   "gamma": (0.85, 1.15), "noise_std": (0.0, 3.0 / 255.0),
                   "blur_sigma": (0.3, 0.7)}
BLUR_P = 0.3
MAX_CAMERA_DRAWS = 10
MIN_VISIBLE_PX = 20  # at 256x256; scaled by image area


def front_camera(dxy, dz: float, dpitch_deg: float, dyaw_deg: float = 0.0,
                 ddist: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
  """Front camera pose from offsets around the nominal pose (copy of eval/ood.py's
  front_camera; with dyaw = ddist = 0 it is exactly sample_variation's construction).
  Translate pos and aim point by dxy, raise pos by dz, orbit pos about the vertical axis
  through the aim point by dyaw, move pos along the view axis by ddist (+ = farther),
  aim at the aim point, then tilt by dpitch about the camera's horizontal axis."""
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


def camera_draws(seed: int, n: int, scale: float = 1.0) -> list[dict]:
  """The first n camera draws of this episode's wide-camera stream (draw k is the k-th
  redraw). ``scale`` shrinks every half-width (the time-box fallback, e.g. 0.7)."""
  rng = np.random.default_rng([int(seed), TAG_WIDE_CAMERA])
  out = []
  for _ in range(n):
    out.append({k: float(rng.uniform(-h * scale, h * scale)) for k, h in WIDE_RANGES.items()})
  return out


def exposure_draw(seed: int, scale: float = 1.0) -> dict:
  """Per-episode exposure parameters (own stream). ``scale`` shrinks each range about 1
  (about 0 for noise and blur)."""
  rng = np.random.default_rng([int(seed), TAG_WIDE_EXPOSURE])

  def around1(lo, hi):
    return 1.0 + scale * (float(rng.uniform(lo, hi)) - 1.0)

  e = {"brightness": around1(*EXPOSURE_RANGES["brightness"]),
       "contrast": around1(*EXPOSURE_RANGES["contrast"]),
       "wb_gain": [around1(*EXPOSURE_RANGES["wb_gain"]) for _ in range(3)],
       "gamma": around1(*EXPOSURE_RANGES["gamma"]),
       "noise_std": scale * float(rng.uniform(*EXPOSURE_RANGES["noise_std"]))}
  blur = rng.uniform() < BLUR_P
  sig = float(rng.uniform(*EXPOSURE_RANGES["blur_sigma"]))  # drawn always (stream alignment)
  e["blur_sigma"] = scale * sig if blur else 0.0
  return e


def with_camera(v: Variation, draw: dict, k: int) -> Variation:
  """Copy of v with its front camera set from one wide draw (k = draw index)."""
  v = copy.deepcopy(v)
  narrow_pos = v.extras.get("wide", {}).get("narrow_camera", {}).get("cam_pos", v.cam_pos)
  dxy = (np.array(narrow_pos) - FRONT_CAM_POS)[:2]  # keep the normal draw's x/y offset
  pos, lookat = front_camera(dxy, draw["dz_m"], draw["dpitch_deg"], draw["dyaw_deg"],
                             draw["ddist_m"])
  wide = v.extras.setdefault("wide", {})
  wide.setdefault("narrow_camera", {"cam_pos": v.cam_pos, "cam_lookat": v.cam_lookat,
                                    "cam_fovy": v.cam_fovy, "cam_dz": v.cam_dz,
                                    "cam_dpitch_deg": v.cam_dpitch_deg})
  v.cam_pos = [float(x) for x in pos]
  v.cam_lookat = [float(x) for x in lookat]
  v.cam_fovy = float(FRONT_CAM_FOVY + draw["dfovy_deg"])
  v.cam_dz = float(draw["dz_m"])
  v.cam_dpitch_deg = float(draw["dpitch_deg"])
  wide["camera"] = {**draw, "fovy_deg": v.cam_fovy, "dxy_m": [float(x) for x in dxy],
                    "draw": int(k), "fallback_narrow": False}
  return v


def with_narrow_camera(v: Variation) -> Variation:
  """Copy of v with the normal (narrow) camera restored; exposure kept. Used when no wide
  draw passes the visibility check."""
  v = copy.deepcopy(v)
  n = v.extras["wide"]["narrow_camera"]
  v.cam_pos, v.cam_lookat, v.cam_fovy = list(n["cam_pos"]), list(n["cam_lookat"]), n["cam_fovy"]
  v.cam_dz, v.cam_dpitch_deg = n["cam_dz"], n["cam_dpitch_deg"]
  v.extras["wide"]["camera"] = {"draw": -1, "fallback_narrow": True}
  return v


def make_wide(v: Variation, scale: float = 1.0) -> Variation:
  """v (from sample_variation) with wide camera draw 0 and the episode's exposure."""
  w = with_camera(v, camera_draws(v.seed, 1, scale)[0], 0)
  w.extras["wide"]["exposure"] = exposure_draw(v.seed, scale)
  w.extras["wide"]["scale"] = float(scale)
  return w


# ---- exposure ---------------------------------------------------------------------------------
def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
  """Separable Gaussian blur (reflect padding) of an HxWxC float image."""
  r = max(1, int(4.0 * sigma + 0.5))
  t = np.arange(-r, r + 1, dtype=np.float32)
  k = np.exp(-0.5 * (t / np.float32(sigma)) ** 2)
  k /= k.sum()
  for ax in (0, 1):
    pad = [(0, 0)] * 3
    pad[ax] = (r, r)
    xp = np.pad(x, pad, mode="symmetric")
    n = x.shape[ax]
    acc = np.zeros_like(x)
    for j, kj in enumerate(k):
      sl = [slice(None)] * 3
      sl[ax] = slice(j, j + n)
      acc += kj * xp[tuple(sl)]
    x = acc
  return x


def apply_exposure(img: np.ndarray, brightness: float = 1.0, contrast: float = 1.0,
                   wb_gain=(1.0, 1.0, 1.0), gamma: float = 1.0, noise_std: float = 0.0,
                   blur_sigma: float = 0.0, rng: np.random.Generator | None = None,
                   **_ignored) -> np.ndarray:
  """Camera-style transform of an HxWx3 uint8 frame, same order and meaning as
  eval/ood.py's apply_exposure: white-balance gains, brightness gain, contrast about
  mid-grey (0.5), gamma (out = in**(1/g)), Gaussian blur, additive Gaussian noise (std in
  [0,1] units); clip; round to uint8. Identity settings return the frame unchanged."""
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
    x = _blur(x, float(blur_sigma))
  if noise_std > 0:
    rng = rng if rng is not None else np.random.default_rng(0)
    x = x + rng.normal(0.0, noise_std, x.shape).astype(np.float32)
  return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def noise_rng(seed: int, cam_index: int) -> np.random.Generator:
  """Per-episode, per-camera noise stream (frames consume it in order)."""
  return np.random.default_rng([int(seed), TAG_WIDE_NOISE, int(cam_index)])


# ---- visibility -------------------------------------------------------------------------------
def visible_pixels(v: Variation, qpos0: np.ndarray, size=(256, 256), object_kind: str = "box",
                   renderer_cache: dict | None = None) -> dict:
  """Render the front camera's segmentation at qpos0; return pixel counts of the cube and
  of the gripper (the gripper body and every body below it: both jaws)."""
  import mujoco

  from .scene import build_episode_scene
  sc = build_episode_scene(v, object_kind=object_kind, render_size=size)
  m = sc.model
  d = mujoco.MjData(m)
  d.qpos[:] = qpos0
  mujoco.mj_forward(m, d)
  # bodies in the gripper subtree
  grip = {sc.grip_body}
  for b in range(m.nbody):
    p, chain = b, []
    while p > 0 and p not in grip:
      chain.append(p)
      p = int(m.body_parentid[p])
    if p in grip:
      grip.update(chain)
  obj_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.obj_body}
  grip_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in grip}
  w, h = size
  r = mujoco.Renderer(m, h, w)
  try:
    r.enable_segmentation_rendering()
    r.update_scene(d, camera="front")
    seg = r.render()
  finally:
    r.close()
  ids, types = seg[..., 0], seg[..., 1]
  is_geom = types == int(mujoco.mjtObj.mjOBJ_GEOM)
  cube = int(np.isin(ids[is_geom], list(obj_geoms)).sum())
  grp = int(np.isin(ids[is_geom], list(grip_geoms)).sum())
  return {"cube_px": cube, "gripper_px": grp}


def min_px(size) -> int:
  return max(1, int(round(MIN_VISIBLE_PX * size[0] * size[1] / (256 * 256))))


def choose_camera(v: Variation, qpos0: np.ndarray, size=(256, 256), object_kind: str = "box",
                  log=None) -> tuple[Variation, dict]:
  """Draw 0, 1, ... of the wide-camera stream until the cube and the gripper are both
  visible in the first frame. Returns the chosen Variation and a record of every try."""
  scale = float(v.extras["wide"].get("scale", 1.0))
  draws = camera_draws(v.seed, MAX_CAMERA_DRAWS, scale)
  thr = min_px(size)
  tries = []
  for k, dr in enumerate(draws):
    vk = with_camera(v, dr, k)
    px = visible_pixels(vk, qpos0, size, object_kind)
    ok = px["cube_px"] >= thr and px["gripper_px"] >= thr
    tries.append({"draw": k, **px, "ok": bool(ok)})
    if ok:
      return vk, {"draws": k + 1, "redraws": k, "fallback_narrow": False, "min_px": thr,
                  "tries": tries}
    if log is not None:
      log(f"[wide] attempt {v.attempt}: camera draw {k} rejected (cube {px['cube_px']} px, "
          f"gripper {px['gripper_px']} px < {thr}); redrawing")
  vn = with_narrow_camera(v)
  px = visible_pixels(vn, qpos0, size, object_kind)
  if log is not None:
    log(f"[wide] attempt {v.attempt}: no visible draw in {MAX_CAMERA_DRAWS}; narrow camera "
        f"kept (cube {px['cube_px']} px, gripper {px['gripper_px']} px)")
  return vn, {"draws": MAX_CAMERA_DRAWS, "redraws": MAX_CAMERA_DRAWS,
              "fallback_narrow": True, "min_px": thr, "tries": tries, "narrow_px": px}
