"""Per-episode random draws ("variations"), deterministic from (base seed, attempt id).

Ranges, and where they come from:
  object start      spec start-range centre +-3 cm in x and y; yaw = spec yaw range widened
                    by +-0.3 rad (R2R2R-style: re-pose the demo's object, keep its path shape)
  object size       spec size +-3 mm, clamped to the 20-50 mm band the task spec validator
                    and the grasp gate cover (smaller cubes squeeze out at the jaw tips)
  object mass       spec mass x 0.61-1.65 (same as the mjlab task's DR)
  friction          object 0.4-1.0, finger pads 0.4-1.4 (the mjlab task's DR, after Squint)
  bin               spec place-range centre +-2 cm; >= spec min_separation from the object
  servos            arm kp and kv x0.5-1.5 per joint; jaw kp 500-2000 N*m/rad (Squint);
                    torque: half the episodes use the 7.4 V STS3215 (1.62-1.91 N*m, the
                    SO-101 BOM part), half the 12 V one (2.94 N*m x0.85-1.0); jaw = 50 % of arm
  look (render)     table colour/texture, cube and bin colour, key-light direction and
                    strength, ambient, headlight (Squint: colour jitter mattered most)
  expert (--variety) operator tempo x0.7-1.3 on all Cartesian speeds, pre-approach height
                    5-10 cm, hover 3-6 cm, side offset 2-6 cm, approach tilt cap 4-16 deg,
                    pauses before close 0.15-0.6 s / before release 0.1-0.6 s / after
                    release 0.3-0.7 s, a 0.2-0.8 s hesitation at the hover point in 35 % of
                    episodes, start pose home +-0.15 rad (instead of +-0.05). Drawn from a
                    separate stream, so every other draw is identical with or without it.
  cameras (render)  front 3/4 camera: height +-8 cm, pitch +-8 deg, x/y +-3 cm, FOV +-3 deg
                    around a nominal pose; the wrist camera is the Menagerie model's own
                    ``wrist_cam`` (fixed to the gripper, not randomised)
  widened (--widen, JEPA-judge negatives only) object mass x0.61-4.0, object friction
                    0.15-1.0, finger friction 0.15-1.4, jaw kp 150-2000, arm torque x0.6-1.0;
                    separate stream, so layout/look/camera match the unwidened scene
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .paths import TaskSpec, rm

PAN_AXIS = np.array([0.0388, 0.0])
SAFE_REACH = (0.08, 0.30)  # m from the pan axis; measured near-vertical band is 0.06-0.32

# Nominal front camera (robot base frame): in front of and to the right of the arm,
# looking back at the workspace centre. Distance ~0.53 m.
FRONT_CAM_POS = np.array([0.56, -0.24, 0.34])
FRONT_CAM_LOOKAT = np.array([0.17, 0.0, 0.03])
FRONT_CAM_FOVY = 45.0

HOME_Q = np.array([0.0, -0.6, 0.4, 1.45, -1.55])  # same home as tests/test_grasp_physics.py


@dataclass
class Variation:
  attempt: int
  seed: int
  # object
  obj_xy: list[float]
  obj_yaw: float
  obj_size: float
  obj_mass: float
  obj_friction: float
  finger_friction: float
  # place
  bin_xy: list[float]
  # servos
  servo_variant: str  # "7v4" | "12v"
  arm_effort: float
  gripper_effort: float
  arm_kp: list[float]  # 5
  arm_kv: list[float]  # 5
  jaw_kp: float
  jaw_kv: float
  # start state
  q_start: list[float]  # 5 arm joints
  gripper_start: float
  # look (render only; drawn for every attempt so seeds stay aligned)
  table_rgb1: list[float]
  table_rgb2: list[float]
  table_texrepeat: float
  object_rgb: list[float]
  bin_rgb: list[float]
  light_dir: list[float]
  light_diffuse: float
  ambient: float
  headlight: float
  # front camera
  cam_pos: list[float]
  cam_lookat: list[float]
  cam_fovy: float
  cam_dz: float
  cam_dpitch_deg: float
  extras: dict = field(default_factory=dict)

  def to_dict(self) -> dict:
    return asdict(self)

  @staticmethod
  def from_dict(d: dict) -> "Variation":
    return Variation(**d)


def episode_seed(base_seed: int, attempt: int) -> int:
  return int(base_seed) * 1_000_003 + int(attempt)


def _hsv(h: float, s: float, v: float) -> list[float]:
  i = int(h * 6) % 6
  f = h * 6 - math.floor(h * 6)
  p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
  r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
  return [float(r), float(g), float(b)]


def _in_reach(xy: np.ndarray) -> bool:
  r = float(np.linalg.norm(xy - PAN_AXIS))
  return SAFE_REACH[0] <= r <= SAFE_REACH[1]


def _pitch_rotate(pos: np.ndarray, lookat: np.ndarray, dpitch: float) -> np.ndarray:
  """Rotate the look-at point about the camera's horizontal axis by dpitch (rad, + = up)."""
  f = lookat - pos
  dist = np.linalg.norm(f)
  f = f / dist
  horiz = np.array([f[0], f[1], 0.0])
  horiz /= np.linalg.norm(horiz)
  elev = math.asin(np.clip(f[2], -1, 1)) + dpitch
  f2 = np.array([horiz[0] * math.cos(elev), horiz[1] * math.cos(elev), math.sin(elev)])
  return pos + f2 * dist


def sample_variation(spec: TaskSpec, base_seed: int, attempt: int,
                     xy_jitter: float = 0.03, bin_jitter: float = 0.02,
                     variety: bool = False, widen: bool = False) -> Variation:
  seed = episode_seed(base_seed, attempt)
  rng = np.random.default_rng(seed)
  o, p = spec.object, spec.place
  obj_c = np.array([(o.x.lo + o.x.hi) / 2, (o.y.lo + o.y.hi) / 2])
  bin_c = np.array([(p.x.lo + p.x.hi) / 2, (p.y.lo + p.y.hi) / 2])
  min_sep = max(p.min_separation, 0.11)
  # Rejection-sample start/bin until both are in reach and far enough apart. These
  # are draws, not rejections: the verifier never sees an infeasible layout.
  for _ in range(1000):
    obj_xy = obj_c + rng.uniform(-xy_jitter, xy_jitter, 2)
    bin_xy = bin_c + rng.uniform(-bin_jitter, bin_jitter, 2)
    if _in_reach(obj_xy) and _in_reach(bin_xy) and np.linalg.norm(obj_xy - bin_xy) >= min_sep:
      break
  else:
    raise RuntimeError("could not sample a feasible object/bin layout from this spec")
  obj_yaw = float(rng.uniform(o.yaw.lo - 0.3, o.yaw.hi + 0.3))
  obj_size = float(np.clip(o.size + rng.uniform(-0.003, 0.003), 0.020, 0.050))
  obj_mass = float(o.mass * rng.uniform(0.61, 1.65))
  obj_friction = float(rng.uniform(0.4, 1.0))
  finger_friction = float(rng.uniform(0.4, 1.4))

  if rng.uniform() < 0.5:
    variant = "7v4"
    arm_effort = float(rng.uniform(rm.STS3215_TORQUE_7V4_6V, 1.91))
  else:
    variant = "12v"
    arm_effort = float(rm.STS3215_TORQUE_12V * rng.uniform(0.85, 1.0))
  gripper_effort = arm_effort * rm.GRIPPER_TORQUE_FRACTION
  arm_kp = (rm.STS3215_KP * rng.uniform(0.5, 1.5, 5)).tolist()
  arm_kv = (rm.STS3215_KV * rng.uniform(0.5, 1.5, 5)).tolist()
  jaw_kp = float(rng.uniform(500.0, 2000.0))
  jaw_kv = float(rm.STS3215_KV * rng.uniform(0.5, 2.0))
  q_start = (HOME_Q + rng.uniform(-0.05, 0.05, 5)).tolist()
  gripper_start = float(rng.uniform(0.0, 0.4))

  # Look.
  base_v = rng.uniform(0.35, 0.9)
  table_hue = rng.uniform(0.0, 1.0)
  table_sat = rng.uniform(0.0, 0.35)
  table_rgb1 = _hsv(table_hue, table_sat, base_v)
  table_rgb2 = _hsv(table_hue, table_sat, float(np.clip(base_v + rng.uniform(-0.12, 0.12), 0.2, 1.0)))
  object_rgb = _hsv(rng.uniform(), rng.uniform(0.5, 0.95), rng.uniform(0.5, 0.95))
  bin_rgb = _hsv(rng.uniform(), rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.9))
  az = rng.uniform(-math.pi, math.pi)
  el = rng.uniform(math.radians(35), math.radians(80))
  light_dir = [-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)]
  light_diffuse = float(rng.uniform(0.35, 0.9))
  ambient = float(rng.uniform(0.08, 0.35))
  headlight = float(rng.uniform(0.15, 0.5))

  # Front camera: randomise height and pitch in a band around the nominal pose.
  dz = float(rng.uniform(-0.08, 0.08))
  dpitch = float(rng.uniform(-8.0, 8.0))
  dxy = rng.uniform(-0.03, 0.03, 2)
  cam_pos = FRONT_CAM_POS + np.array([dxy[0], dxy[1], dz])
  # Keep the nominal aim point under a pure translation, then tilt by dpitch.
  lookat = FRONT_CAM_LOOKAT + np.array([dxy[0], dxy[1], 0.0])
  lookat = _pitch_rotate(cam_pos, lookat, math.radians(dpitch))
  cam_fovy = float(FRONT_CAM_FOVY + rng.uniform(-3.0, 3.0))

  extras: dict = {}
  if variety:
    from .planner import HOVER_PAUSE_P, VARIETY_RANGES
    r2 = np.random.default_rng([seed, 7])  # separate stream: the draws above are unchanged
    plan = {k: float(r2.uniform(lo, hi)) for k, (lo, hi) in VARIETY_RANGES.items()}
    if r2.uniform() >= HOVER_PAUSE_P:
      plan["hover_pause_s"] = 0.0
    extras["plan"] = plan
    q_start = (HOME_Q + r2.uniform(-0.15, 0.15, 5)).tolist()

  if widen:
    # JEPA-judge negatives: physics ranges widened past the dataset's DR, drawn from a
    # separate stream so the layout, look and camera stay identical to the unwidened
    # scene with the same seed (same scene, harder physics). Nothing is welded or moved:
    # these are ordinary parameter draws, judged by the same verifier.
    r3 = np.random.default_rng([seed, 11])
    obj_mass = float(o.mass * r3.uniform(0.61, 4.0))
    obj_friction = float(r3.uniform(0.15, 1.0))
    finger_friction = float(r3.uniform(0.15, 1.4))
    jaw_kp = float(r3.uniform(150.0, 2000.0))
    arm_effort = float(arm_effort * r3.uniform(0.6, 1.0))
    gripper_effort = arm_effort * rm.GRIPPER_TORQUE_FRACTION
    extras["widen"] = True

  return Variation(
    attempt=int(attempt), seed=int(seed),
    obj_xy=[float(v) for v in obj_xy], obj_yaw=obj_yaw, obj_size=obj_size,
    obj_mass=obj_mass, obj_friction=obj_friction, finger_friction=finger_friction,
    bin_xy=[float(v) for v in bin_xy],
    servo_variant=variant, arm_effort=arm_effort, gripper_effort=gripper_effort,
    arm_kp=arm_kp, arm_kv=arm_kv, jaw_kp=jaw_kp, jaw_kv=jaw_kv,
    q_start=q_start, gripper_start=gripper_start,
    table_rgb1=table_rgb1, table_rgb2=table_rgb2,
    table_texrepeat=float(rng.uniform(2.0, 12.0)),
    object_rgb=object_rgb, bin_rgb=bin_rgb, light_dir=[float(v) for v in light_dir],
    light_diffuse=light_diffuse, ambient=ambient, headlight=headlight,
    cam_pos=[float(v) for v in cam_pos], cam_lookat=[float(v) for v in lookat],
    cam_fovy=cam_fovy, cam_dz=dz, cam_dpitch_deg=dpitch, extras=extras,
  )
