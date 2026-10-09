"""Hook: let datagen use a mesh object (made by ``object3d/mesh_to_mjcf.py``) instead of the cube.

The cube stays the default. A task spec opts in with:

    "object": {"kind": "mesh", "mjcf": "path/to/object_dir_or_object.xml", "mass": 0.094,
               "scale_mult": 1.0, "start_pose_range": {...}}

``mjcf`` is resolved relative to the spec file. ``mass`` defaults to the mass in the
object's report.json. ``scale_mult`` (default 1) rescales the whole object about its
bounding-box centre; ``object3d/validate_object.py`` uses it for the +-3 % / +-5 % test.

How it plugs in (nothing else in datagen changes behaviour for box/cylinder specs):
  * :func:`load_spec` parses the JSON with the task module's own validator (the object is
    presented to it as a nominal 30 mm box, because ``task/`` only knows box|cylinder), then
    sets ``spec.object.kind = "mesh"`` and attaches ``spec.mesh`` (a :class:`MeshObject`).
  * ``scene.build_episode_scene(..., mesh=spec.mesh)`` attaches the mesh body instead of the
    primitive and exposes its half extents; the planner and the verifier read the object's
    size from there instead of ``Variation.obj_size``.
  * The per-episode random draws are unchanged, so cube and mesh runs with the same seed get
    the same pose, servo and friction draws (paired comparison). The cube-size draw is simply
    not used by a mesh object.

The mesh body follows the same physics-honesty rules: one free body named "object" with a
free joint, no actuator, no equality constraint; only its collision pieces (contype != 0)
touch anything.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .paths import TaskSpec, load_task_spec
from .paths import mj_scene  # noqa: F401  (re-exported for callers)

NOMINAL_BOX_SIZE = 0.030  # what the task-module validator is shown in place of a mesh


@dataclass
class MeshObject:
  xml: Path  # object.xml written by object3d/mesh_to_mjcf.py
  half_extents_m: np.ndarray  # (3,) bounding-box half sizes about the body origin, scale 1
  mass: float  # kg, nominal (the spec's mass DR multiplies it per episode)
  scale_mult: float = 1.0
  name: str = "mesh"

  @staticmethod
  def from_path(path: str | Path, scale_mult: float = 1.0, mass: float | None = None) -> "MeshObject":
    p = Path(path)
    xml = p / "object.xml" if p.is_dir() else p
    if not xml.is_file():
      raise FileNotFoundError(f"mesh object MJCF not found: {xml}")
    rep_path = xml.with_name("report.json")
    rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.is_file() else {}
    half = rep.get("half_extents_m")
    if half is None:  # no report: measure the compiled collision geoms
      half = _measure_half_extents(xml)
    return MeshObject(xml=xml.resolve(), half_extents_m=np.asarray(half, float),
                      mass=float(mass if mass is not None else rep.get("mass_kg", 0.05)),
                      scale_mult=float(scale_mult), name=rep.get("name", xml.parent.name))

  @property
  def half(self) -> np.ndarray:
    """Half extents (m) at this object's scale."""
    return self.half_extents_m * self.scale_mult

  def with_scale(self, scale_mult: float) -> "MeshObject":
    o = copy.copy(self)
    o.scale_mult = float(scale_mult)
    return o

  def object_spec(self, mass: float, friction: tuple[float, float, float],
                  rgba: tuple[float, float, float, float] | None = None,
                  with_visual: bool = True) -> mujoco.MjSpec:
    """MjSpec of the object body, ready for ``spec.attach``. File paths are made absolute
    (the parent model lives elsewhere); masses are rescaled so the body weighs ``mass``;
    friction goes on every collision piece. ``with_visual=False`` drops the drawn-only
    mesh (it has no mass and no contacts, so physics is identical; compiles faster)."""
    spec = mujoco.MjSpec.from_file(str(self.xml))
    base = self.xml.parent
    for m in spec.meshes:
      if m.file:
        m.file = str((base / m.file).resolve())
      m.scale = np.asarray(m.scale, float) * self.scale_mult
    for t in spec.textures:
      if t.file:
        t.file = str((base / t.file).resolve())
    body = spec.body("object")
    col = [g for g in body.geoms if g.contype != 0 or g.conaffinity != 0]
    if not col:
      raise ValueError(f"{self.xml}: no collision geoms in body 'object'")
    total = float(sum(g.mass for g in col))
    for g in col:
      g.mass = g.mass * mass / total
      g.friction = list(friction)
    vis = [g for g in body.geoms if g.contype == 0 and g.conaffinity == 0]
    if not with_visual:
      for g in vis:
        spec.delete(g)
      for m in list(spec.meshes):
        if m.name.endswith("_visual"):
          spec.delete(m)
    elif rgba is not None:
      for g in vis:
        if not g.material:
          g.rgba = list(rgba)
    return spec


def _measure_half_extents(xml: Path) -> list[float]:
  m = mujoco.MjModel.from_xml_path(str(xml))
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  pts = []
  for g in range(m.ngeom):
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]:
      mid = m.geom_dataid[g]
      a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
      v = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
      pts.append(v - d.xpos[m.geom_bodyid[g]])
  p = np.concatenate(pts)
  return (np.abs(p).max(0)).tolist()


def load_spec(path: str | Path) -> TaskSpec:
  """Like ``load_task_spec``, plus object kind "mesh". Box/cylinder specs go straight to
  the task module's loader (unchanged behaviour)."""
  path = Path(path)
  d = json.loads(path.read_text(encoding="utf-8"))
  o = d.get("object", {}) or {}
  if o.get("kind") != "mesh":
    return load_task_spec(path)
  if "mjcf" not in o:
    raise ValueError("object.kind 'mesh' needs object.mjcf (object.xml or its folder)")
  mjcf = Path(o["mjcf"])
  if not mjcf.is_absolute():
    mjcf = (path.parent / mjcf).resolve()
  mesh = MeshObject.from_path(mjcf, scale_mult=float(o.get("scale_mult", 1.0)),
                              mass=o.get("mass"))
  return spec_with_mesh(d, mesh, str(path))


def spec_with_mesh(d: dict, mesh: MeshObject, path: str | None = None) -> TaskSpec:
  """Parse spec dict ``d`` with ``mesh`` as its object (used by load_spec and the
  validation harness)."""
  from .paths import TASK_ROOT  # noqa: F401  (task module already on sys.path)
  from so101_pickplace.task_spec import parse_task_spec

  d2 = copy.deepcopy(d)
  o = dict(d2.get("object", {}) or {})
  for k in ("mjcf", "scale_mult"):
    o.pop(k, None)
  o.update(kind="box", size=NOMINAL_BOX_SIZE, mass=mesh.mass)
  d2["object"] = o
  spec = parse_task_spec(d2, path=path)
  spec.object.kind = "mesh"
  spec.object.size = float(2 * min(mesh.half[:2]))
  spec.object.height = float(2 * mesh.half[2])
  spec.mesh = mesh  # type: ignore[attr-defined]
  return spec


def spec_mesh(spec: TaskSpec) -> MeshObject | None:
  return getattr(spec, "mesh", None)
