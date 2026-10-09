"""Tests for the real-to-sim data engine (CPU MuJoCo only; no GL needed).

  python -I -m pytest -q -p no:cacheprovider manip/datagen/tests/test_datagen.py   (~40 s)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

DATAGEN = Path(__file__).resolve().parents[1]
if str(DATAGEN) not in sys.path:
  sys.path.insert(0, str(DATAGEN))

from r2s import scene as S  # noqa: E402
from r2s.lerobot_io import EpisodeData, write_dataset  # noqa: E402
from r2s.paths import load_task_spec  # noqa: E402
from r2s.planner import VARIETY_RANGES, PlanConfig, plan_config_for  # noqa: E402
from r2s.render import _Encoder  # noqa: E402
from r2s.rollout import run_episode  # noqa: E402
from r2s.variations import sample_variation  # noqa: E402

SUGAR = DATAGEN.parent / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"
SYNTH = DATAGEN.parent / "demo" / "outputs" / "synthetic" / "so101_task_spec.json"


def test_variations_are_deterministic_and_in_range():
  spec = load_task_spec(SUGAR)
  a, b = sample_variation(spec, 7, 3), sample_variation(spec, 7, 3)
  assert a.to_dict() == b.to_dict()
  assert sample_variation(spec, 7, 4).to_dict() != a.to_dict()
  for i in range(50):
    v = sample_variation(spec, 0, i)
    assert 0.020 <= v.obj_size <= 0.050
    assert np.linalg.norm(np.array(v.obj_xy) - np.array(v.bin_xy)) >= 0.11
    assert abs(v.cam_dz) <= 0.08 and abs(v.cam_dpitch_deg) <= 8.0


def test_variety_only_adds_plan_params_and_start_pose():
  spec = load_task_spec(SUGAR)
  for i in range(20):
    plain, var = sample_variation(spec, 0, i).to_dict(), sample_variation(spec, 0, i, variety=True).to_dict()
    changed = {k for k in plain if plain[k] != var[k]}
    assert changed == {"q_start", "extras"}, changed  # layout, physics, look, camera unchanged
    p = var["extras"]["plan"]
    for k, (lo, hi) in VARIETY_RANGES.items():
      assert p[k] == 0.0 if k == "hover_pause_s" and p[k] == 0.0 else lo <= p[k] <= hi, (k, p[k])
    assert np.abs(np.array(var["q_start"]) - np.array(plain["q_start"])).max() < 0.15 + 0.05 + 1e-9
  v = sample_variation(spec, 0, 1, variety=True)
  cfg, base = plan_config_for(v), PlanConfig()
  k = v.extras["plan"]["speed_scale"]
  assert abs(cfg.line_speed - base.line_speed * k) < 1e-12
  assert cfg.pre_close_s == v.extras["plan"]["pre_close_s"]
  assert plan_config_for(sample_variation(spec, 0, 1)) == base  # no variety -> defaults


def test_varied_episode_is_physical():
  spec = load_task_spec(SUGAR)
  acc = [run_episode(spec, sample_variation(spec, 0, i, variety=True)).accepted for i in range(4)]
  assert sum(acc) >= 3, acc


def test_episode_model_has_no_weld_latch_or_mocap():
  spec = load_task_spec(SUGAR)
  sc = S.build_episode_scene(sample_variation(spec, 0, 0))
  m = sc.model
  assert m.neq == 0 and m.nmocap == 0
  j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
  assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
  for a in range(m.nu):
    assert m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] != j
  assert {m.camera(i).name for i in range(m.ncam)} == {"front", "wrist_cam"}


def test_accepted_episode_is_physical():
  spec = load_task_spec(SUGAR)
  r = run_episode(spec, sample_variation(spec, 0, 1))  # attempt 1 is accepted at seed 0
  assert r.accepted, r.reasons
  s = r.stats
  assert s["contacts"]["carry_steps"] > 300
  assert s["contacts"]["carry_both_jaws"] == s["contacts"]["carry_steps"]
  assert s["contacts"]["carry_foreign"] == 0 and s["contacts"]["self"] == 0
  assert s["max_obj_lift_mm"] > 40
  dv = np.abs(np.diff(r.action, axis=0)) * S.FPS
  assert dv[:, :5].max() <= 0.5 + 1e-5 and dv[:, 5].max() <= 2.0 + 1e-5
  assert len(r.state) == len(r.action) == len(r.env_state) == len(r.phases)


def test_negative_control_jaw_never_closes_is_rejected():
  spec = load_task_spec(SUGAR)
  r = run_episode(spec, sample_variation(spec, 0, 1), cfg=PlanConfig(close_jaw=False))
  assert not r.accepted
  assert r.primary_reason == "grasp_failed", r.reasons
  assert r.stats["max_obj_lift_mm"] < 5.0


def test_writer_with_fake_video_validates():
  """Exercise the LeRobot writer + validator video path with synthetic frames."""
  from validate_dataset import validate
  tmp = Path(tempfile.mkdtemp(prefix="datagen_test_"))
  try:
    eps = []
    rng = np.random.default_rng(0)
    for e, T in enumerate((12, 17)):
      vids = {}
      for k in ("observation.images.front", "observation.images.wrist"):
        p = tmp / f"ep{e}_{k.split('.')[-1]}.mp4"
        enc = _Encoder(p, (64, 48), "h264")
        for t in range(T):
          enc.add(np.full((48, 64, 3), (t * 10) % 255, np.uint8))
        enc.close()
        vids[k] = p
      st = {"sum": [1.0] * 3, "sumsq": [1.0] * 3, "min": [0.0] * 3, "max": [1.0] * 3, "n": 4}
      q = np.cumsum(rng.uniform(-0.01, 0.01, (T, 6)), 0).astype(np.float32)
      eps.append(EpisodeData(state=q, action=q, env_state=np.zeros((T, 7), np.float32),
                             task="pick up the cube and place it in the bin", videos=vids,
                             image_stats={k: st for k in vids}, meta={"attempt": e}))
    root = tmp / "ds"
    info = write_dataset(root, eps, fps=30, video_keys=list(vids), video_size=(64, 48),
                         video_codec="h264", robot_type="so101_follower", extra_meta={})
    assert info["total_frames"] == 29
    c = validate(root)
    assert not c.fail, c.fail
    assert json.loads((root / "meta" / "info.json").read_text())["codebase_version"] == "v3.0"
  finally:
    shutil.rmtree(tmp, ignore_errors=True)
