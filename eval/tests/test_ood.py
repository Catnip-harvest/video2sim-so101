"""OOD ladder tests (CPU, no GL for most; ~20 s).

  python -m pytest -q -p no:cacheprovider eval/tests/test_ood.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

import ood as O  # noqa: E402
import sim_eval as E  # noqa: E402
from r2s.paths import load_task_spec  # noqa: E402
from r2s.variations import FRONT_CAM_LOOKAT, FRONT_CAM_POS, sample_variation  # noqa: E402

SPEC = load_task_spec(E.SPEC)
HARNESS_RUN = Path("/work/outputs_large/eval/policy_final_n200_seed3000/episodes.jsonl")
IDS = range(50)

RENDER_FIELDS = {"camera": {"cam_pos", "cam_lookat", "cam_fovy", "cam_dz", "cam_dpitch_deg"},
                 "colour": {"object_rgb", "bin_rgb", "table_rgb1", "table_rgb2", "light_diffuse",
                            "ambient", "headlight"}}


def _dump(v) -> str:
  return json.dumps(v.to_dict(), sort_keys=True)


def test_none_is_identity_seed4000():
  for i in IDS:
    v = sample_variation(SPEC, 4000, i, variety=True)
    ref = _dump(v)
    out = O.apply_ood(v, "none")
    assert out is v
    assert _dump(out) == ref == _dump(sample_variation(SPEC, 4000, i, variety=True))


@pytest.mark.skipif(not HARNESS_RUN.exists(), reason="P1.1 harness run not on this box")
def test_none_matches_harness_run_byte_for_byte():
  """The Variations the P1.1 harness (eval/harness, before --ood) wrote to episodes.jsonl are
  reproduced exactly by this branch with --ood none."""
  n = 0
  for line in HARNESS_RUN.read_text().splitlines():
    r = json.loads(line)
    v = O.apply_ood(sample_variation(SPEC, r["seed"], r["attempt"], variety=True), "none")
    assert json.dumps(v.to_dict()) == json.dumps(r["variation"]), r["attempt"]
    n += 1
  assert n >= 50


def test_rungs_only_touch_their_fields():
  for rung in O.RUNGS[1:]:
    allowed = {"extras"}
    if rung.startswith("camera") or rung == "combo":
      allowed |= RENDER_FIELDS["camera"]
    if rung in ("colour", "combo"):
      allowed |= RENDER_FIELDS["colour"]
    for i in range(10):
      v0 = sample_variation(SPEC, 4000, i, variety=True)
      ref = v0.to_dict()
      v1 = O.apply_ood(v0, rung)
      assert v0.to_dict() == ref, "apply_ood must not mutate its input"
      d1 = v1.to_dict()
      changed = {k for k in ref if ref[k] != d1[k]}
      assert changed <= allowed, (rung, changed - allowed)
      assert d1["extras"]["ood"]["rung"] == rung
      assert d1["extras"].get("plan") == ref["extras"].get("plan")  # expert tempo unchanged
      assert O.apply_ood(sample_variation(SPEC, 4000, i, variety=True), rung).to_dict() == d1  # deterministic


def test_front_camera_reproduces_training_construction():
  for i in IDS:
    v = sample_variation(SPEC, 4000, i, variety=True)
    dxy = (np.array(v.cam_pos) - FRONT_CAM_POS)[:2]
    pos, look = O.front_camera(dxy, v.cam_dz, v.cam_dpitch_deg)
    assert np.allclose(pos, v.cam_pos, atol=1e-12) and np.allclose(look, v.cam_lookat, atol=1e-12)


def _angles(pos, look):
  f = np.asarray(look) - np.asarray(pos)
  f /= np.linalg.norm(f)
  return math.degrees(math.asin(f[2])), math.degrees(math.atan2(f[1], f[0]))


def test_camera_rungs_are_outside_the_training_band():
  for i in IDS:
    v0 = sample_variation(SPEC, 4000, i, variety=True)
    dxy = (np.array(v0.cam_pos) - FRONT_CAM_POS)[:2]
    aim = FRONT_CAM_LOOKAT + np.array([dxy[0], dxy[1], 0.0])
    c = O.apply_ood(v0, "camera")
    rec = c.extras["ood"]["camera"]
    assert 0.12 <= abs(rec["dz_m"]) <= 0.16 and 12 <= abs(rec["dpitch_deg"]) <= 15
    assert c.cam_fovy == 51.0 and abs(c.cam_pos[2] - FRONT_CAM_POS[2]) >= 0.12 - 1e-9
    h = O.apply_ood(v0, "camera_height")
    assert abs(h.cam_dz) >= 0.12 and h.cam_dpitch_deg == v0.cam_dpitch_deg and h.cam_fovy == v0.cam_fovy
    p = O.apply_ood(v0, "camera_pitch")
    assert abs(p.cam_dpitch_deg) >= 12 and p.cam_pos == v0.cam_pos
    # yaw: the camera orbits the aim point; distance and height to it unchanged
    y = O.apply_ood(v0, "camera_yaw")
    a0 = math.atan2(v0.cam_pos[1] - aim[1], v0.cam_pos[0] - aim[0])
    a1 = math.atan2(y.cam_pos[1] - aim[1], y.cam_pos[0] - aim[0])
    dyaw = math.degrees((a1 - a0 + math.pi) % (2 * math.pi) - math.pi)
    assert 10 - 1e-6 <= abs(dyaw) <= 15 + 1e-6
    assert abs(np.linalg.norm(np.array(y.cam_pos) - aim) - np.linalg.norm(np.array(v0.cam_pos) - aim)) < 1e-9
    # distance: moves along the view axis
    dd = O.apply_ood(v0, "camera_distance")
    r0 = np.linalg.norm(np.array(v0.cam_pos) - aim)
    r1 = np.linalg.norm(np.array(dd.cam_pos) - aim)
    assert 0.08 - 1e-9 <= abs(r1 - r0) <= 0.12 + 1e-9
    assert np.allclose(_angles(dd.cam_pos, dd.cam_lookat), _angles(v0.cam_pos, v0.cam_lookat), atol=1e-6)


def test_signs_are_mixed():
  recs = [O.apply_ood(sample_variation(SPEC, 4000, i, variety=True), "camera").extras["ood"]["camera"]
          for i in IDS]
  for k in ("dz_m", "dpitch_deg"):
    s = sum(r[k] > 0 for r in recs)
    assert 10 <= s <= 40, (k, s)


def test_combo_is_camera_plus_colour():
  for i in range(10):
    v0 = sample_variation(SPEC, 4000, i, variety=True)
    cb, cam, col = (O.apply_ood(v0, r).to_dict() for r in ("combo", "camera", "colour"))
    for k in RENDER_FIELDS["camera"]:
      assert cb[k] == cam[k]
    for k in RENDER_FIELDS["colour"]:
      assert cb[k] == col[k]


def test_exposure_draws_and_transform():
  kinds = []
  for i in IDS:
    e = O.apply_ood(sample_variation(SPEC, 4000, i, variety=True), "exposure").extras["ood"]["exposure"]
    kinds.append(e["kind"])
    assert e["brightness"] in (0.5, 1.0, 1.6) and e["contrast"] in (0.6, 1.0, 1.4)
    assert all(g in (0.8, 1.0, 1.2) for g in e["wb_gain"]) and e["wb_gain"][1] == 1.0
  assert set(kinds) == {"brightness", "contrast", "white_balance"}
  img = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
  assert O.apply_exposure(img) is img
  half = O.apply_exposure(img, brightness=0.5)
  assert np.abs(half.astype(int) - np.round(img / 2).astype(int)).max() <= 1
  wb = O.apply_exposure(np.full((2, 2, 3), 100, np.uint8), wb_gain=(1.2, 1.0, 0.8))
  assert wb[0, 0].tolist() == [120, 100, 80]
  c = O.apply_exposure(np.array([[[64, 128, 191]]], np.uint8), contrast=1.4)
  assert c[0, 0, 0] < 64 and c[0, 0, 2] > 191


def test_colour_is_outside_training_palette():
  import colorsys
  for i in range(10):
    v = O.apply_ood(sample_variation(SPEC, 4000, i, variety=True), "colour")
    _, s, val = colorsys.rgb_to_hsv(*v.object_rgb)
    assert s > 0.95 and val > 0.95  # training cube: s, v in 0.5-0.95
    assert colorsys.rgb_to_hsv(*v.bin_rgb)[2] < 0.3  # training bin value >= 0.3
    assert colorsys.rgb_to_hsv(*v.table_rgb1)[1] > 0.35  # training table saturation <= 0.35
    assert (v.light_diffuse < 0.35 and v.ambient < 0.08 and v.headlight < 0.15) or \
           (v.light_diffuse > 0.9 and v.ambient > 0.35 and v.headlight > 0.5)


def test_distractor_rung_and_wrong_object_clause():
  import mujoco
  from r2s.verify_policy import PhaseFreeMonitor
  for i in range(20):
    v = O.apply_ood(sample_variation(SPEC, 4000, i, variety=True), "distractor")
    fb = v.extras["free_bodies"]
    assert 1 <= len(fb) <= 2 and v.extras["ood"]["distractor"]["n"] == len(fb)
    assert all(b["shape"] in ("cylinder", "sphere") for b in fb)
  # negative control: a distractor resting in the bin -> task failure wrong_object_in_bin;
  # the same body outside the bin -> no such clause.
  v = O.apply_ood(sample_variation(SPEC, 4000, 0, variety=True), "distractor")
  for inside in (True, False):
    w = json.loads(json.dumps(v.to_dict()))
    if inside:
      w["extras"]["free_bodies"][0]["xy"] = list(w["bin_xy"])
    from r2s.variations import Variation
    w = Variation.from_dict(w)
    sc = E.setup_scene(SPEC, w)
    mon = PhaseFreeMonitor(sc)
    for _ in range(int(0.5 / sc.model.opt.timestep)):
      mujoco.mj_step(sc.model, sc.data)
      mon.step(w)
    vd = mon.verdict(SPEC, w)
    assert ("wrong_object_in_bin" in vd["reasons"]) == inside
    assert ("wrong_object_in_bin" in vd["task_reasons"]) == inside
    assert vd["stats"]["extra_bodies"][0]["in_bin"] == inside
