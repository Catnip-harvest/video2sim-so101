"""Fast checks for calib/overlay_gate.py (run inside the sprint container, MUJOCO_GL=osmesa)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from calib.overlay_gate import cam_frame, iou, real_arm_mask  # noqa: E402


def test_camera_looks_at_lookat():
  p = [0.2, -0.1, 0.5, 30.0, 60.0, 15.0, 50.0]
  pos, R = cam_frame(p)
  assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
  ray = np.array([0.2, -0.1, 0.0]) - pos
  assert np.isclose(np.linalg.norm(ray), 0.5)
  assert np.allclose(ray / 0.5, -R[:, 2])  # MuJoCo cameras look down their -z axis


def test_real_mask_keeps_arm_drops_wire_and_cube():
  img = np.full((480, 640, 3), 190, np.uint8)       # light table
  img[100:300, 100:250] = 30                         # black arm part
  img[400:403, 0:640] = 20                           # 3 px black wire
  img[100:200, 400:500] = (20, 60, 40)               # dark side of the green cube
  m = real_arm_mask(img)
  assert m[110:290, 110:240].all()
  assert not m[400:403].any()
  assert not m[100:200, 400:500].any()


def test_iou():
  a = np.zeros((10, 10), bool); a[:5] = True
  b = np.zeros((10, 10), bool); b[:, :5] = True
  assert np.isclose(iou(a, b), 25 / 75)


def test_sim_mask_renders_arm():
  from calib.overlay_gate import SimArm
  sim = SimArm(160, 120)
  sim.set_camera([0.15, 0.0, 0.6, 180.0, 60.0, 0.0, 60.0])
  m = sim.mask(np.zeros(6), 160, 120)
  assert 0.01 < m.mean() < 0.5
  assert np.allclose(sim.d.cam_xpos[sim.cam], cam_frame([0.15, 0.0, 0.6, 180.0, 60.0, 0.0, 60.0])[0])
