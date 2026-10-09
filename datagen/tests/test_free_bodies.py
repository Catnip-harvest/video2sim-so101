"""Extra free bodies (distractors / extra cubes): CPU MuJoCo only, ~10 s.

  python -m pytest -q -p no:cacheprovider datagen/tests/test_free_bodies.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

DATAGEN = Path(__file__).resolve().parents[1]
if str(DATAGEN) not in sys.path:
  sys.path.insert(0, str(DATAGEN))

from r2s import scene as S  # noqa: E402
from r2s.free_bodies import SHAPES, half_height, sample_free_bodies  # noqa: E402
from r2s.paths import load_task_spec, mj_scene  # noqa: E402
from r2s.variations import PAN_AXIS, Variation, sample_variation  # noqa: E402

SUGAR = DATAGEN.parent / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"
SPEC = load_task_spec(SUGAR)


def _with(v: Variation, bodies: list[dict]) -> Variation:
  v = copy.deepcopy(v)
  v.extras = {**(v.extras or {}), "free_bodies": bodies}
  return v


def _reset(sc, v: Variation) -> None:
  """Arm at the episode's start pose, servos holding it (as the harness/rollout do)."""
  m, d = sc.model, sc.data
  mujoco.mj_resetData(m, d)
  d.qpos[sc.arm_qadr] = v.q_start
  d.ctrl[sc.arm_act] = v.q_start
  d.qpos[sc.g_qadr] = d.ctrl[sc.g_act] = v.gripper_start
  mujoco.mj_forward(m, d)


def _settle(sc, seconds: float) -> None:
  for _ in range(int(seconds / sc.model.opt.timestep)):
    mujoco.mj_step(sc.model, sc.data)


def test_no_extras_builds_the_same_model():
  v = sample_variation(SPEC, 4000, 0, variety=True)
  a = S.build_episode_scene(v, object_kind=SPEC.object.kind)
  b = S.build_episode_scene(_with(v, []), object_kind=SPEC.object.kind)
  assert a.model.nq == b.model.nq and a.model.ngeom == b.model.ngeom and a.extra_bodies == ()
  assert not np.any(a.geom_cat == S.CAT_EXTRA)


def test_every_shape_is_a_free_body_resting_on_the_table():
  v = sample_variation(SPEC, 4000, 1, variety=True)
  rng = np.random.default_rng(0)
  for shape in SHAPES:
    bodies = sample_free_bodies(v, rng, 2, shapes=(shape,))
    sc = S.build_episode_scene(_with(v, bodies), object_kind=SPEC.object.kind)
    m, d = sc.model, sc.data
    assert m.neq == 0 and m.nmocap == 0  # no weld, no mocap
    assert len(sc.extra_bodies) == 2
    for bid, b in zip(sc.extra_bodies, bodies):
      j = m.body_jntadr[bid]
      assert m.body_jntnum[bid] == 1 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
      g = m.body_geomadr[bid]
      assert sc.geom_cat[g] == S.CAT_EXTRA
      assert m.geom_contype[g] == m.geom_contype[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")]
    _reset(sc, v)
    _settle(sc, 0.3)
    p0 = np.array([d.xpos[b] for b in sc.extra_bodies])
    _settle(sc, 2.0)
    p1 = np.array([d.xpos[b] for b in sc.extra_bodies])
    for k, b in enumerate(bodies):  # resting on the table, not sinking, not rolling away
      assert abs(p1[k, 2] - half_height(b)) < 0.002, (shape, p1[k, 2], half_height(b))
      assert np.linalg.norm(p1[k] - p0[k]) < 0.001, (shape, p1[k] - p0[k])


def test_bodies_collide_physically():
  """Drop one body onto another: it must come to rest ON it (contact), not pass through."""
  v = sample_variation(SPEC, 4000, 2, variety=True)
  rng = np.random.default_rng(1)
  base = sample_free_bodies(v, rng, 1, shapes=("box",))[0]
  top = dict(base, name="extra_top", shape="box", size=0.02)
  sc = S.build_episode_scene(_with(v, [base, top]), object_kind=SPEC.object.kind)
  m, d = sc.model, sc.data
  _reset(sc, v)
  j = m.body_jntadr[sc.extra_bodies[1]]
  qa = m.jnt_qposadr[j]
  d.qpos[qa + 2] = base["size"] + 0.05  # test-only initial state, 5 cm above the base body
  mujoco.mj_forward(m, d)
  _settle(sc, 1.5)
  z = d.xpos[sc.extra_bodies[1], 2]
  assert abs(z - (base["size"] + 0.01)) < 0.004, z


def test_sampler_constraints_and_determinism():
  bin_half = mj_scene.BIN_INNER_HALF + mj_scene.BIN_WALL_THICKNESS
  for i in range(100):
    v = sample_variation(SPEC, 4000, i, variety=True)
    a = sample_free_bodies(v, np.random.default_rng([v.seed, 104]), 2)
    b = sample_free_bodies(v, np.random.default_rng([v.seed, 104]), 2)
    assert json.dumps(a) == json.dumps(b)
    for k, x in enumerate(a):
      p = np.array(x["xy"])
      assert np.linalg.norm(p - np.array(v.obj_xy)) >= 0.06
      d = np.maximum(np.abs(p - np.array(v.bin_xy)) - bin_half, 0.0)
      assert np.linalg.norm(d) >= 0.06
      assert 0.10 <= np.linalg.norm(p - PAN_AXIS) <= 0.28
      assert 0.020 <= x["size"] <= 0.035 and x["shape"] in ("cylinder", "sphere")
      for y in a[:k]:
        assert np.linalg.norm(p - np.array(y["xy"])) >= 0.05
