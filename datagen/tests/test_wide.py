"""Tests for --wide (r2s/wide.py). CPU MuJoCo; the GL tests skip without OpenGL.

  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -I -m pytest -q -p no:cacheprovider \
      datagen/tests/test_wide.py

The base-code video check (test_wide_off_videos_identical_to_base) runs only when
BASE_DATAGEN points at a checkout of the code before --wide (e.g. /work/wt-datagen).
"""

from __future__ import annotations

import filecmp
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

DATAGEN = Path(__file__).resolve().parents[1]
if str(DATAGEN) not in sys.path:
  sys.path.insert(0, str(DATAGEN))

from r2s import wide as W  # noqa: E402
from r2s.paths import load_task_spec  # noqa: E402
from r2s.rollout import run_episode  # noqa: E402
from r2s.variations import FRONT_CAM_FOVY, FRONT_CAM_LOOKAT, FRONT_CAM_POS, sample_variation  # noqa: E402

SPEC_PATH = DATAGEN.parent / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"
SPEC = load_task_spec(SPEC_PATH)
FIXTURE = DATAGEN / "tests" / "fixtures" / "variations_seed1_base.json"
CAM_KEYS = {"cam_pos", "cam_lookat", "cam_fovy", "cam_dz", "cam_dpitch_deg"}


def _gl() -> bool:
  from r2s.render import gl_available
  return gl_available()[0]


def test_wide_off_variations_identical_to_base_code():
  """Without --wide, sample_variation returns exactly what the pre---wide code returned."""
  base = json.loads(FIXTURE.read_text())
  for i in range(30):
    assert json.loads(json.dumps(sample_variation(SPEC, 1, i).to_dict())) == base["plain"][i]
    assert json.loads(json.dumps(sample_variation(SPEC, 1, i, variety=True).to_dict())) == base["variety"][i]


def test_wide_only_changes_camera_and_adds_extras():
  for i in range(40):
    plain = sample_variation(SPEC, 1, i, variety=True).to_dict()
    wide = sample_variation(SPEC, 1, i, variety=True, wide=True).to_dict()
    changed = {k for k in plain if plain[k] != wide[k]}
    assert changed <= CAM_KEYS | {"extras"}, changed
    assert wide["extras"]["plan"] == plain["extras"]["plan"]
    assert set(wide["extras"]) == {"plan", "wide"}
    n = wide["extras"]["wide"]["narrow_camera"]
    assert n["cam_pos"] == plain["cam_pos"] and n["cam_fovy"] == plain["cam_fovy"]


def test_wide_draws_in_range_and_deterministic():
  for i in range(200):
    v = sample_variation(SPEC, 1, i, wide=True)
    c, e = v.extras["wide"]["camera"], v.extras["wide"]["exposure"]
    for k, h in W.WIDE_RANGES.items():
      assert abs(c[k]) <= h
    assert abs(v.cam_fovy - FRONT_CAM_FOVY) <= 8.0
    for k in ("brightness", "contrast", "gamma"):
      lo, hi = W.EXPOSURE_RANGES[k]
      assert lo <= e[k] <= hi
    assert all(0.9 <= g <= 1.1 for g in e["wb_gain"])
    assert 0.0 <= e["noise_std"] <= 3 / 255 and (e["blur_sigma"] == 0 or 0.3 <= e["blur_sigma"] <= 0.7)
  assert sample_variation(SPEC, 1, 5, wide=True).to_dict() == sample_variation(SPEC, 1, 5, wide=True).to_dict()
  # the draws really span the ranges
  cs = np.array([[sample_variation(SPEC, 1, i, wide=True).extras["wide"]["camera"][k]
                  for k in W.WIDE_RANGES] for i in range(200)])
  assert np.all(cs.max(0) > 0.8 * np.array(list(W.WIDE_RANGES.values())))
  blur = [sample_variation(SPEC, 1, i, wide=True).extras["wide"]["exposure"]["blur_sigma"] > 0 for i in range(200)]
  assert 0.2 < np.mean(blur) < 0.4


def test_wide_camera_geometry():
  """Height raises the camera, distance moves it along the view axis, yaw keeps the range."""
  aim = FRONT_CAM_LOOKAT
  p0, _ = W.front_camera([0, 0], 0, 0)
  assert np.allclose(p0, FRONT_CAM_POS)
  p, _ = W.front_camera([0, 0], 0.15, 0)
  assert abs(p[2] - FRONT_CAM_POS[2] - 0.15) < 1e-12
  p, _ = W.front_camera([0, 0], 0, 0, 0, 0.10)
  assert abs(np.linalg.norm(p - aim) - np.linalg.norm(FRONT_CAM_POS - aim) - 0.10) < 1e-9
  p, la = W.front_camera([0, 0], 0, 0, 15.0)
  assert abs(np.linalg.norm(p - aim) - np.linalg.norm(FRONT_CAM_POS - aim)) < 1e-9
  assert np.allclose(la, aim)  # zero pitch: still aimed at the same point
  a = np.degrees(np.arctan2(p[1] - aim[1], p[0] - aim[0]) - np.arctan2(FRONT_CAM_POS[1] - aim[1], FRONT_CAM_POS[0] - aim[0]))
  assert abs(a - 15.0) < 1e-9


def test_redraws_come_from_one_stream():
  d10 = W.camera_draws(1_000_003 * 1 + 7, 10)
  assert W.camera_draws(1_000_003 * 1 + 7, 3) == d10[:3]
  assert len({json.dumps(d) for d in d10}) == 10
  v = sample_variation(SPEC, 1, 7, wide=True)
  assert v.extras["wide"]["camera"]["dz_m"] == d10[0]["dz_m"]
  v3 = W.with_camera(v, d10[3], 3)
  assert v3.extras["wide"]["narrow_camera"] == v.extras["wide"]["narrow_camera"]
  assert v3.extras["wide"]["camera"]["dxy_m"] == v.extras["wide"]["camera"]["dxy_m"]


def test_exposure_transform():
  img = np.random.default_rng(0).integers(0, 256, (16, 16, 3)).astype(np.uint8)
  assert W.apply_exposure(img) is img
  half = W.apply_exposure(np.full((2, 2, 3), 100, np.uint8), brightness=0.75)
  assert np.all(half == 75)
  wb = W.apply_exposure(np.full((2, 2, 3), 100, np.uint8), wb_gain=(1.1, 1.0, 0.9))
  assert tuple(wb[0, 0]) == (110, 100, 90)
  blurred = W.apply_exposure(img, blur_sigma=0.7)
  assert blurred.std() < img.std()
  flat = np.full((32, 32, 3), 128, np.uint8)
  assert np.array_equal(W.apply_exposure(flat, blur_sigma=0.7), flat)  # blur keeps a flat image
  n1 = W.apply_exposure(flat, noise_std=3 / 255, rng=W.noise_rng(5, 0))
  n2 = W.apply_exposure(flat, noise_std=3 / 255, rng=W.noise_rng(5, 0))
  assert np.array_equal(n1, n2) and 0.5 < (n1.astype(float) - 128).std() < 5


def test_wide_physics_identical_to_plain():
  """--wide changes no physics: same accept verdict and bit-identical arrays."""
  for i in (0, 1):
    a = run_episode(SPEC, sample_variation(SPEC, 1, i, variety=True))
    b = run_episode(SPEC, sample_variation(SPEC, 1, i, variety=True, wide=True))
    assert a.accepted == b.accepted and a.reasons == b.reasons
    for k in ("state", "action", "env_state", "qpos_full"):
      assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert a.stats == b.stats


@pytest.mark.skipif(not _gl(), reason="needs OpenGL (MUJOCO_GL=osmesa)")
def test_visibility_check_and_redraw():
  v = sample_variation(SPEC, 1, 0, variety=True, wide=True)
  r = run_episode(SPEC, v)
  px = W.visible_pixels(W.with_narrow_camera(v), r.qpos_full[0])
  assert px["cube_px"] >= 20 and px["gripper_px"] >= 20, px
  # a camera pitched far up sees neither
  away = W.with_camera(v, {"dz_m": 0.0, "dpitch_deg": 60.0, "dyaw_deg": 0.0, "ddist_m": 0.0,
                           "dfovy_deg": 0.0}, 0)
  px = W.visible_pixels(away, r.qpos_full[0])
  assert px["cube_px"] == 0 and px["gripper_px"] == 0, px
  chosen, rec = W.choose_camera(v, r.qpos_full[0])
  assert rec["tries"][-1]["ok"] or rec["fallback_narrow"]
  assert chosen.extras["wide"]["camera"].get("draw") == rec["draws"] - 1 or rec["fallback_narrow"]
  # the chosen camera changes nothing but the camera
  assert {k for k in v.to_dict() if v.to_dict()[k] != chosen.to_dict()[k]} <= CAM_KEYS | {"extras"}


def _gen(code: Path, out: Path, extra: list[str]) -> None:
  env = dict(os.environ, MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="1")
  cmd = [sys.executable, str(code / "datagen" / "generate.py"), "--spec", str(SPEC_PATH),
         "--n", "1", "--render", "--render-quality", "fast", "--record-hz", "10", "--variety",
         "--size", "128x128", "--codec", "h264", "--seed", "1", "--workers", "1",
         "--no-dataset", "--out", str(out), *extra]
  subprocess.run(cmd, check=True, env=env, capture_output=True)


@pytest.mark.skipif(not os.environ.get("BASE_DATAGEN") or not _gl(),
                    reason="set BASE_DATAGEN=<checkout before --wide> and MUJOCO_GL=osmesa")
def test_wide_off_videos_identical_to_base():
  base = Path(os.environ["BASE_DATAGEN"])
  with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    _gen(base, t / "base", [])
    _gen(DATAGEN.parent, t / "new", [])
    _gen(DATAGEN.parent, t / "wide", ["--wide"])
    files = sorted(p.name for p in (t / "base" / "staging").iterdir())
    assert any(f.endswith(".mp4") for f in files)
    for f in files:
      if f.endswith(".mp4"):  # byte-identical videos
        assert filecmp.cmp(t / "base" / "staging" / f, t / "new" / "staging" / f, shallow=False), f
      if f.endswith(".npz"):  # npz zip headers carry a timestamp: compare the arrays
        zb, zn = np.load(t / "base" / "staging" / f), np.load(t / "new" / "staging" / f)
        assert zb.files == zn.files and all(np.array_equal(zb[k], zn[k]) for k in zb.files), f
      if f.endswith(".json"):
        jb, jn = (json.loads((t / d / "staging" / f).read_text()) for d in ("base", "new"))
        assert jb["variation"] == jn["variation"] and jb["image_stats"] == jn["image_stats"]
    zb = np.load(next((t / "base" / "staging").glob("*.npz")))
    zw = np.load(next((t / "wide" / "staging").glob("*.npz")))
    for k in zb.files:
      assert np.array_equal(zb[k], zw[k]), k
    fb = next((t / "base" / "staging").glob("*.front.mp4"))
    assert not filecmp.cmp(fb, t / "wide" / "staging" / fb.name, shallow=False)
