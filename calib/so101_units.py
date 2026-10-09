"""The one place where SO-101 joint values change units or convention.

Three conventions meet in this project:

* ``sim``: MuJoCo Menagerie ``robotstudio_so101`` joint angles in RADIANS, zero = the
  MJCF zero (mid-range, from so101_new_calib.xml). The gripper is a hinge in radians,
  range -0.1745..1.7453 (-10..100 deg). Our datagen writes this.
* ``lerobot``: LeRobot >= PR #777 with ``use_degrees=True`` (the friend's dataset
  ``MinhDuk/final_dataset_annotated``). Arm joints in DEGREES, 0 = middle of the
  calibrated range; gripper 0..100 = percent of calibrated jaw travel (0 = closed).
* ``molmoact2``: the frame ``allenai/MolmoAct2-SO100_101`` was trained in, i.e. LeRobot's
  pre-PR #777 calibration. Per lerobot/policies/molmoact2/processor_molmoact2.py and
  https://huggingface.co/docs/lerobot/backwardcomp:
      shoulder_lift_old = 90 - shoulder_lift_new,  elbow_flex_old = elbow_flex_new + 90,
  every other joint unchanged. (Gripper scale in that frame is NOT verified.)

``lerobot`` <-> ``sim`` is per arm and is what calibration (calib/overlay_gate.py) checks:
    sim_rad = deg2rad(sign * lerobot_deg + offset_deg)              (arm joints)
    sim_rad = deg2rad(g_lo + (g_hi - g_lo) * lerobot_pct / 100)     (gripper)
Defaults: identity on pan/lift/elbow/wrist_flex; wrist_roll offset -90 deg (so-frame's
bridge.py measured sim = real - 90 on their arm); gripper 0..100 -> -10..100 deg
(leisaac's linear map). These defaults are hypotheses until the overlay gate passes.

All functions take and return arrays shaped [..., 6] in the order of JOINTS.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LEROBOT_NAMES = [f"{j}.pos" for j in JOINTS]

# lerobot (new calibration) -> molmoact2 (old calibration): model = SIGNS * new + OFFSETS.
# These are exactly the values the LeRobot molmoact2 policy takes as joint_signs/joint_offsets.
MOLMOACT2_SIGNS = (1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
MOLMOACT2_OFFSETS = (0.0, 90.0, 90.0, 0.0, 0.0, 0.0)


@dataclass
class ArmCalib:
  """lerobot-degrees <-> sim-radians mapping for one physical arm."""
  sign: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0])
  offset_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, -90.0])
  gripper_deg_at_0: float = -10.0    # sim gripper angle (deg) when lerobot gripper = 0
  gripper_deg_at_100: float = 100.0  # ... and when it is 100
  source: str = "defaults (unverified hypotheses, see module docstring)"

  def __post_init__(self):
    if len(self.sign) != 5 or len(self.offset_deg) != 5:
      raise ValueError("sign and offset_deg need 5 values (arm joints, no gripper)")
    if any(s not in (1.0, -1.0) for s in self.sign):
      raise ValueError(f"signs must be +-1, got {self.sign}")
    if self.gripper_deg_at_100 == self.gripper_deg_at_0:
      raise ValueError("gripper map is degenerate")

  def save(self, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(self), indent=2))

  @classmethod
  def load(cls, path: str | Path) -> "ArmCalib":
    return cls(**json.loads(Path(path).read_text()))


def _as(x) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64)
  if x.shape[-1] != 6:
    raise ValueError(f"expected last dim 6 ({JOINTS}), got shape {x.shape}")
  return x


def lerobot_to_sim(x, cal: ArmCalib | None = None) -> np.ndarray:
  cal = cal or ArmCalib()
  x = _as(x)
  out = np.empty_like(x)
  out[..., :5] = np.deg2rad(np.asarray(cal.sign) * x[..., :5] + np.asarray(cal.offset_deg))
  g0, g1 = cal.gripper_deg_at_0, cal.gripper_deg_at_100
  out[..., 5] = np.deg2rad(g0 + (g1 - g0) * x[..., 5] / 100.0)
  return out


def sim_to_lerobot(q, cal: ArmCalib | None = None) -> np.ndarray:
  cal = cal or ArmCalib()
  q = _as(q)
  out = np.empty_like(q)
  # sign is +-1, so dividing equals multiplying by it
  out[..., :5] = (np.rad2deg(q[..., :5]) - np.asarray(cal.offset_deg)) * np.asarray(cal.sign)
  g0, g1 = cal.gripper_deg_at_0, cal.gripper_deg_at_100
  out[..., 5] = (np.rad2deg(q[..., 5]) - g0) * 100.0 / (g1 - g0)
  return out


def lerobot_to_molmoact2(x) -> np.ndarray:
  x = _as(x)
  return np.asarray(MOLMOACT2_SIGNS) * x + np.asarray(MOLMOACT2_OFFSETS)


def molmoact2_to_lerobot(x) -> np.ndarray:
  x = _as(x)
  return np.asarray(MOLMOACT2_SIGNS) * (x - np.asarray(MOLMOACT2_OFFSETS))


def subsample(n_frames: int, src_hz: int, dst_hz: int) -> np.ndarray:
  """Frame indices that turn a src_hz recording into dst_hz (every k-th frame, from 0).
  Used for EVERY source so all training data shares one frame rate."""
  if src_hz % dst_hz:
    raise ValueError(f"{src_hz} Hz is not a multiple of {dst_hz} Hz")
  return np.arange(0, n_frames, src_hz // dst_hz)
