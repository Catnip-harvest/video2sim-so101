"""Tests for object3d (mesh -> MJCF, validation harness) and its datagen hook.

  python -I -m pytest -q -p no:cacheprovider object3d/tests        (~30 s, CPU only)

Uses only synthetic meshes made with trimesh, so it needs no downloads.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh

OBJECT3D = Path(__file__).resolve().parents[1]
REPO = OBJECT3D.parent
for p in (OBJECT3D, REPO / "datagen"):
  if str(p) not in sys.path:
    sys.path.insert(0, str(p))

import mesh_to_mjcf as M  # noqa: E402
import validate_object as V  # noqa: E402

SUGAR = REPO / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"


def _box_obj(tmp: Path, extents=(30.0, 30.0, 30.0), yaw_deg=0.0, name="box.obj") -> Path:
  """A box in MILLIMETRES (so scaling to metres is exercised), turned by yaw_deg."""
  m = trimesh.creation.box(extents=extents)
  m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(yaw_deg), [0, 0, 1]))
  m.apply_translation([5.0, -3.0, extents[2] / 2])  # off-centre, sitting on z = 0
  p = tmp / name
  m.export(p)
  return p


def test_footprint_alignment_recovers_the_turn(tmp_path):
  m = trimesh.creation.box(extents=(60, 30, 20))
  m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(25), [0, 0, 1]))
  yaw = M.footprint_yaw(np.asarray(m.vertices))
  v = np.asarray(m.vertices) @ M.rot_z(yaw).T
  ext = v.max(0) - v.min(0)
  assert np.allclose(ext, [60, 30, 20], atol=1e-6)  # longer side on x


def test_cap_holes_closes_an_open_cup():
  m = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=24)
  top = m.triangles_center[:, 2] > 0.99
  open_cup = trimesh.Trimesh(m.vertices, m.faces[~top], process=False)
  assert not open_cup.is_watertight
  fixed, info = M.repair(np.asarray(open_cup.vertices), np.asarray(open_cup.faces), 1e-9)
  assert info["watertight"] and info["boundary_loops_capped"] == 1
  assert fixed.volume == pytest.approx(math.pi * 2.0, rel=0.02)


def test_convert_box_scale_mass_and_mjcf(tmp_path):
  src = _box_obj(tmp_path, extents=(30, 20, 10), yaw_deg=30)
  rep = M.convert(src, tmp_path / "out", mass=0.05, target_size=0.030, target_dim="x",
                  backend="hull")
  assert rep["scale_factor"] == pytest.approx(0.001, rel=1e-6)  # mm -> m
  assert np.allclose(rep["extents_m"], [0.030, 0.020, 0.010], atol=1e-6)
  assert np.allclose(np.asarray(rep["bbox_m"]).mean(0), 0, atol=1e-9)  # origin = bbox centre
  assert rep["watertight"] and rep["n_hulls"] == 1 and rep["inertia_mode"] == "convex"
  assert rep["mesh_volume_m3"] == pytest.approx(6e-6, rel=1e-6)
  m = mujoco.MjModel.from_xml_path(str(tmp_path / "out" / "object.xml"))
  b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
  assert m.body_mass[b] == pytest.approx(0.05)
  # solid box inertia: m (b^2 + c^2) / 12 about x, etc.
  want = 0.05 / 12 * np.array([0.02**2 + 0.01**2, 0.03**2 + 0.01**2, 0.03**2 + 0.02**2])
  assert np.allclose(np.sort(m.body_inertia[b]), np.sort(want), rtol=1e-3)
  vis = [g for g in range(m.ngeom) if m.geom(g).name == "object_visual"][0]
  assert m.geom_contype[vis] == 0 and m.geom_conaffinity[vis] == 0 and m.geom_group[vis] == 1
  j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
  assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE


def test_shell_mode_and_explicit_scale(tmp_path):
  src = _box_obj(tmp_path)
  rep = M.convert(src, tmp_path / "out", mass=0.02, scale=0.001, inertia="shell", backend="hull")
  assert rep["inertia_mode"] == "shell" and rep["scale_factor"] == 0.001
  m = mujoco.MjModel.from_xml_path(str(tmp_path / "out" / "object.xml"))
  b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
  # thin-walled cube: I = 5/18 m a^2 (vs 1/6 m a^2 solid)
  assert m.body_inertia[b][0] == pytest.approx(5 / 18 * 0.02 * 0.03**2, rel=1e-3)


def test_glb_is_turned_z_up(tmp_path):
  m = trimesh.creation.box(extents=(0.03, 0.08, 0.02))  # glTF: y is up, so height = 0.08
  p = tmp_path / "tall.glb"
  m.export(p)
  rep = M.convert(p, tmp_path / "out", mass=0.02, scale=1.0, backend="hull", align=False)
  assert rep["up_axis"] == "y"
  assert rep["extents_m"][2] == pytest.approx(0.08, abs=1e-6)


def test_hollow_object_decomposes_into_several_pieces(tmp_path):
  if M.available_backends()["vhacd"] != "ok" and M.available_backends()["coacd"] != "ok":
    pytest.skip("no convex-decomposition library loads on this machine")
  ring = trimesh.creation.annulus(r_min=0.03, r_max=0.04, height=0.05)
  p = tmp_path / "ring.obj"
  ring.export(p)
  backend = "vhacd" if M.available_backends()["vhacd"] == "ok" else "coacd"
  rep = M.convert(p, tmp_path / "out", mass=0.05, scale=1.0, backend=backend, max_hulls=16,
                  vhacd_resolution=100_000)
  assert rep["n_hulls"] >= 4
  # the hole stays a hole: the centre is inside no collision piece
  from scipy.spatial import Delaunay
  for f in sorted((tmp_path / "out").glob("collision_*.obj")):
    v = np.array([[float(x) for x in ln.split()[1:4]] for ln in f.read_text().splitlines()
                  if ln.startswith("v ")])
    assert Delaunay(v).find_simplex([[0.0, 0.0, 0.0]])[0] < 0


def test_wilson_interval():
  lo, hi = V.wilson(0, 30)
  assert lo == 0 and hi == pytest.approx(0.1135, abs=1e-3)
  lo, hi = V.wilson(30, 30)
  assert hi == 1 and lo == pytest.approx(0.8865, abs=1e-3)


@pytest.fixture(scope="module")
def mesh_box_30(tmp_path_factory):
  tmp = tmp_path_factory.mktemp("mb")
  src = _box_obj(tmp)
  M.convert(src, tmp / "box30", mass=0.025, target_size=0.030, target_dim="x", backend="hull")
  return tmp / "box30"


def test_settle_mesh_box_passes(mesh_box_30):
  r = V.settle_test(V.ObjectSource(str(mesh_box_30)))
  assert r["passed"], r


def test_datagen_hook_builds_a_weld_free_mesh_scene(mesh_box_30):
  from r2s import scene as S
  from r2s.mesh_object import spec_with_mesh, MeshObject
  from r2s.variations import sample_variation

  d = json.loads(SUGAR.read_text(encoding="utf-8"))
  mesh = MeshObject.from_path(mesh_box_30)
  spec = spec_with_mesh(d, mesh, str(SUGAR))
  assert spec.object.kind == "mesh" and spec.mesh is mesh
  v = sample_variation(spec, 0, 0)
  sc = S.build_episode_scene(v, mesh=spec.mesh, with_visual=False)
  m = sc.model
  assert m.neq == 0 and m.nmocap == 0
  assert np.allclose(sc.obj_half_extents, 0.015, atol=1e-6)
  obj_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == sc.obj_body]
  assert obj_geoms and all(sc.geom_cat[g] == S.CAT_OBJECT for g in obj_geoms)
  assert m.body_subtreemass[sc.obj_body] == pytest.approx(v.obj_mass)
  for a in range(m.nu):
    assert m.actuator_trnid[a, 0] != mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
  # scale_mult rescales the object about its centre
  big = mesh.with_scale(1.05)
  sc2 = S.build_episode_scene(v, mesh=big, with_visual=False)
  assert np.allclose(sc2.obj_half_extents, 0.015 * 1.05, atol=1e-6)


def test_spec_file_with_kind_mesh_loads(tmp_path, mesh_box_30):
  from r2s.mesh_object import load_spec

  d = json.loads(SUGAR.read_text(encoding="utf-8"))
  d["object"]["kind"] = "mesh"
  d["object"]["mjcf"] = str(mesh_box_30)
  d["object"]["mass"] = 0.03
  p = tmp_path / "mesh_spec.json"
  p.write_text(json.dumps(d), encoding="utf-8")
  spec = load_spec(p)
  assert spec.object.kind == "mesh" and spec.object.mass == 0.03 and spec.mesh.mass == 0.03
  assert load_spec(SUGAR).object.kind == "box"  # box specs unchanged


def test_mesh_cube_episode_is_accepted_like_the_primitive(mesh_box_30):
  """Same seed and draw that the primitive-cube test accepts (seed 0, attempt 1), with the
  30 mm cube coming through the mesh pipeline instead: the full verifier accepts it too."""
  from r2s.mesh_object import MeshObject, spec_with_mesh
  from r2s.rollout import run_episode
  from r2s.variations import sample_variation

  d = json.loads(SUGAR.read_text(encoding="utf-8"))
  spec = spec_with_mesh(d, MeshObject.from_path(mesh_box_30), str(SUGAR))
  v = sample_variation(spec, 0, 1)
  r = run_episode(spec, v, keep_arrays=False)
  assert r.accepted, (r.reasons, r.plan_info.get("error"))
  assert r.stats["max_obj_lift_mm"] >= 20
  assert v.extras["mesh"]["half_extents_m"] == pytest.approx([0.015] * 3, abs=1e-6)
