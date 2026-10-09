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
                     variety: bool = False) -> Variation:
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


# ---------------------------------------------------------------------------------------
# P3-A-lite: named cube among three (language-conditioned multi-task scene).
#
# Three cubes of the same size band; the instruction names one by colour. The planner's
# "object" body is always the NAMED cube in expert data; the other two are extra free
# bodies (free_bodies.py, shape "box"). The three positions are drawn as one exchangeable
# set and the named cube takes a uniformly random slot, so position says nothing about
# which cube is named: only the colour (i.e. the instruction) does.
# ---------------------------------------------------------------------------------------

NAMED_COLOURS = {  # base RGB; each episode scales brightness by 0.85-1.0
  "red": (0.85, 0.12, 0.10),
  "green": (0.12, 0.70, 0.18),
  "blue": (0.12, 0.25, 0.90),
  "yellow": (0.92, 0.82, 0.10),
}
TRAIN_TARGETS = ("red", "green", "blue")  # yellow is NEVER the target in training
TRAIN_TEMPLATES = ("put the {c} cube in the tray", "place the {c} block into the tray")
HELDOUT_TEMPLATES = ("move the {c} one to the tray", "{c} cube goes in the tray, please")
MT_LAYOUTS = {
  # name: (reach band from the pan axis (m), min centre-to-centre gap, min centre to bin outer wall)
  "default": ((0.12, 0.26), 0.06, 0.06),
  "narrow": ((0.16, 0.22), 0.09, 0.06),  # Gate A kill fallback
}


def multitask_info(v: "Variation") -> dict | None:
  return (v.extras or {}).get("multitask")


def instruction_for(colour: str, template: str) -> str:
  return template.format(c=colour)


def sample_multitask_variation(spec: TaskSpec, base_seed: int, attempt: int,
                               variety: bool = True, target_colour: str | None = None,
                               template: str | None = None, layout: str = "default",
                               allow_yellow_target: bool = False,
                               distractor_colours: tuple | None = None) -> Variation:
  """A normal (variety) Variation, then re-laid-out as a three-cube scene from a separate
  stream ([seed, 11]): the bin, servos, look, camera and expert-style draws are the
  single-cube ones. Target colour: ``target_colour`` or TRAIN_TARGETS[attempt % 3]
  (balanced). Distractor colours: two of the remaining named colours, at random.
  Instruction: ``template`` or a random TRAIN_TEMPLATES entry. The random stream is consumed
  identically whatever ``target_colour`` / ``template`` / ``distractor_colours`` are, so the
  same (seed, attempt) always gives the same positions, slot, sizes and shades (eval rows
  that change only the instruction see the same physical scene)."""
  from .free_bodies import _dist_to_square
  v = sample_variation(spec, base_seed, attempt, variety=variety)
  r3 = np.random.default_rng([v.seed, 11])
  reach, min_gap, min_bin = MT_LAYOUTS[layout]
  binc = np.asarray(v.bin_xy, float)
  bin_half = rm_bin_outer_half()
  pts: list[np.ndarray] = []
  for _ in range(20000):
    if len(pts) == 3:
      break
    r = r3.uniform(*reach)
    a = math.radians(r3.uniform(-80.0, 80.0))
    p = PAN_AXIS + r * np.array([math.cos(a), math.sin(a)])
    if _dist_to_square(p, binc, bin_half) < min_bin:
      continue
    if not _in_view(v, np.array([p[0], p[1], 0.011])):
      continue
    if any(np.linalg.norm(p - q) < min_gap for q in pts):
      continue
    pts.append(p)
  if len(pts) < 3:
    raise RuntimeError("could not place three cubes with this layout")
  if target_colour is None:
    target_colour = TRAIN_TARGETS[attempt % len(TRAIN_TARGETS)]
  if target_colour not in NAMED_COLOURS:
    raise ValueError(f"unknown colour {target_colour!r}")
  if target_colour == "yellow" and not allow_yellow_target:
    raise ValueError("yellow is never a training target (eval only: allow_yellow_target=True)")
  others = [c for c in NAMED_COLOURS if c != target_colour]
  distr = [others[i] for i in r3.permutation(len(others))[:2]]
  if distractor_colours is not None:
    distr = list(distractor_colours)
    if len(distr) != 2 or len(set(distr) | {target_colour}) != 3 or not set(distr) <= set(NAMED_COLOURS):
      raise ValueError(f"bad distractor colours {distractor_colours!r} for target {target_colour!r}")
  slot = int(r3.integers(3))  # which of the three positions is the named cube
  order = [slot] + [i for i in range(3) if i != slot]
  tpl_draw = TRAIN_TEMPLATES[int(r3.integers(len(TRAIN_TEMPLATES)))]
  if template is None:
    template = tpl_draw
  shade = r3.uniform(0.85, 1.0, 3)

  def rgb(c, k):
    return [float(x * shade[k]) for x in NAMED_COLOURS[c]]

  bodies = []
  for k, (c, pi) in enumerate(zip(distr, order[1:])):
    size = float(np.clip(spec.object.size + r3.uniform(-0.003, 0.003), 0.020, 0.050))
    bodies.append({"name": f"extra_{k}", "shape": "box", "size": size, "height": size,
                   "mass": float(spec.object.mass * r3.uniform(0.61, 1.65)),
                   "friction": float(r3.uniform(0.4, 1.0)), "rgb": rgb(c, k + 1),
                   "xy": [float(x) for x in pts[pi]],
                   "yaw": float(r3.uniform(-math.pi, math.pi))})
  v.obj_xy = [float(x) for x in pts[order[0]]]
  v.object_rgb = rgb(target_colour, 0)
  # Tray: low saturation (greys / browns), so no tray reads as a named colour. Drawn last,
  # so every draw above is unchanged by it (render only, no physics).
  v.bin_rgb = _hsv(float(r3.uniform()), float(r3.uniform(0.0, 0.25)), float(r3.uniform(0.3, 0.85)))
  v.extras = {**v.extras, "free_bodies": bodies, "multitask": {
    "target_colour": target_colour,
    "colours": {"object": target_colour, **{b["name"]: c for b, c in zip(bodies, distr)}},
    "instruction": instruction_for(target_colour, template), "template": template,
    "target_slot": slot, "layout": layout}}
  return v


def retarget(v: "Variation", colour: str, template: str) -> "Variation":
  """Same physical scene, another instruction (instruction-swap eval): the named colour must
  be one of the scene's cubes. The planner's "object" body is unchanged, so for a colour on
  an extra body the expert verifier would (correctly) call the expert's motion wrong."""
  import copy
  w = copy.deepcopy(v)
  mt = w.extras["multitask"]
  if colour not in mt["colours"].values():
    raise ValueError(f"{colour!r} is not in this scene ({mt['colours']})")
  mt.update({"target_colour": colour, "template": template,
             "instruction": instruction_for(colour, template)})
  return w


CAM_VIEW_MARGIN = 0.85  # cube centre within the central 85 % of the front image (both axes)


def _in_view(v: "Variation", p: np.ndarray, margin: float = CAM_VIEW_MARGIN) -> bool:
  """Pinhole projection into this episode's front camera (cam_head; square image, vertical
  FOV v.cam_fovy): True if p lands inside the central ``margin`` of the image."""
  pos = np.asarray(v.cam_pos, float)
  f = np.asarray(v.cam_lookat, float) - pos
  f /= np.linalg.norm(f)
  x = np.cross(f, np.array([0.0, 0.0, 1.0]))
  x /= np.linalg.norm(x)
  y = np.cross(x, f)
  r = p - pos
  zc = float(r @ f)
  if zc <= 0:
    return False
  t = math.tan(math.radians(v.cam_fovy) / 2)
  return abs(float(r @ x) / zc / t) <= margin and abs(float(r @ y) / zc / t) <= margin


def rm_bin_outer_half() -> float:
  from .paths import mj_scene
  return mj_scene.BIN_INNER_HALF + mj_scene.BIN_WALL_THICKNESS


def named_body(v: "Variation") -> str:
  """Body name ("object" or "extra_k") that carries the instruction's colour."""
  mt = multitask_info(v)
  if mt is None:
    return "object"
  for name, c in mt["colours"].items():
    if c == mt["target_colour"]:
      return name
  raise ValueError("no body has the target colour")
