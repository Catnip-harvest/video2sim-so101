"""Sim side of the camera fit: Menagerie SO-101 + one free camera -> binary robot masks.

Camera parameters (7 numbers, robot base frame, metres / radians / degrees):
  x, y, z      camera position
  yaw          rotation about world z
  tilt_a       rotation about the camera's own image-x axis, applied after yaw
  tilt_b       rotation about the camera's own image-y axis
  fovy         vertical field of view in degrees
With yaw = tilt_a = tilt_b = 0 the camera looks straight down (-z) with image-up = world +x.
This chart has no singularity near "looking down", unlike azimuth/elevation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SO101_XML = REPO / "third_party/menagerie_so101/robotstudio_so101/so101.xml"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAM_NAMES = ("x", "y", "z", "yaw", "tilt_a", "tilt_b", "fovy")


def _rx(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# camera axes (columns: image-right x, image-up y, backward z) when looking straight down,
# image-up = world +x, image-right = world -y
R_DOWN = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]).T @ np.eye(3)
R_DOWN = np.column_stack([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def cam_pose(p) -> tuple[np.ndarray, np.ndarray]:
  """7 params -> (position [3], R [3x3] with columns = camera x, y, z axes in the base frame)."""
  p = np.asarray(p, dtype=np.float64)
  R = _rz(p[3]) @ R_DOWN @ _rx(p[4]) @ _ry(p[5])
  return p[:3].copy(), R


def project(p, pts: np.ndarray, w: int = 640, h: int = 480) -> np.ndarray:
  """Pinhole projection (MuJoCo convention) of base-frame points [n,3] -> pixel (u, v) [n,2]."""
  pos, R = cam_pose(p)
  c = (np.asarray(pts) - pos) @ R  # camera coords
  f = 0.5 * h / np.tan(np.deg2rad(p[6]) / 2)
  u = w / 2 + f * c[:, 0] / -c[:, 2]
  v = h / 2 - f * c[:, 1] / -c[:, 2]
  return np.stack([u, v], 1)


def backproject_to_plane(p, uv: np.ndarray, z0: float = 0.0, w: int = 640, h: int = 480) -> np.ndarray:
  """Pixels [n,2] -> base-frame points on the horizontal plane z = z0 [n,3]."""
  pos, R = cam_pose(p)
  f = 0.5 * h / np.tan(np.deg2rad(p[6]) / 2)
  uv = np.asarray(uv, dtype=np.float64)
  d_cam = np.stack([(uv[:, 0] - w / 2) / f, -(uv[:, 1] - h / 2) / f, -np.ones(len(uv))], 1)
  d = d_cam @ R.T
  t = (z0 - pos[2]) / d[:, 2]
  return pos + t[:, None] * d


class SimArm:
  def __init__(self, xml: str | Path = SO101_XML):
    import mujoco
    self.mj = mujoco
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.worldbody.add_camera(name="fit", pos=[0, 0, 1], fovy=60)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480
    self.m = spec.compile()
    self.d = mujoco.MjData(self.m)
    self.cam = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, "fit")
    self.qadr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS]
    self.renderers = {}

  def _r(self, w, h):
    if (w, h) not in self.renderers:
      r = self.mj.Renderer(self.m, h, w)
      r.enable_segmentation_rendering()
      self.renderers[(w, h)] = r
    return self.renderers[(w, h)]

  def set_camera(self, p) -> None:
    pos, R = cam_pose(p)
    q = np.zeros(4)
    self.mj.mju_mat2Quat(q, R.flatten())
    self.m.cam_pos[self.cam], self.m.cam_quat[self.cam], self.m.cam_fovy[self.cam] = pos, q, p[6]

  def mask(self, q_sim: np.ndarray, w: int = 160, h: int = 120) -> np.ndarray:
    self.d.qpos[self.qadr] = q_sim
    self.mj.mj_kinematics(self.m, self.d)
    self.mj.mj_camlight(self.m, self.d)
    r = self._r(w, h)
    r.update_scene(self.d, camera=self.cam)
    seg = r.render()
    return seg[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM)

  def body_points(self, q_sim) -> np.ndarray:
    self.d.qpos[self.qadr] = q_sim
    self.mj.mj_kinematics(self.m, self.d)
    return self.d.xpos[1:].copy()


def lowpoly_xml(cache: str | Path = "/work/outputs_large/friendrig/so101_lowpoly", target_faces: int = 1500) -> Path:
  """Copy of the Menagerie SO-101 with every STL decimated to about `target_faces` triangles
  (fast-simplification quadric decimation, aggressiveness 10). Only used to make the camera FIT fast
  (OSMesa cost is per triangle: 348k faces -> 25 ms per mask at any size). Final IoU numbers are
  always re-measured with the full meshes."""
  import shutil
  import fast_simplification
  import trimesh
  cache = Path(cache)
  xml = cache / "so101.xml"
  if xml.exists():
    return xml
  src = SO101_XML.parent
  (cache / "assets").mkdir(parents=True, exist_ok=True)
  for f in (src / "assets").glob("*.stl"):
    m = trimesh.load_mesh(f)
    if len(m.faces) > target_faces:
      v, fc = fast_simplification.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces),
                                           target_count=target_faces, agg=10)
      m = trimesh.Trimesh(v, fc, process=False)
    m.export(cache / "assets" / f.name)
  shutil.copy(src / "so101.xml", xml)
  return xml


class PointArm:
  """Fast silhouette without OpenGL: the visible robot surface sampled as points (area-weighted),
  moved with forward kinematics, projected with the pinhole model, splatted into a grid and
  closed. A silhouette is a union, so no depth test is needed. ~1 ms per mask at 160x120 versus
  25 ms for OSMesa on the 348k-triangle Menagerie meshes. Used only inside the optimiser; every
  reported IoU is re-measured with MuJoCo's own renderer (SimArm, full meshes)."""

  def __init__(self, n_points: int = 60000, seed: int = 0, xml: str | Path = SO101_XML, bodies=None):
    import mujoco
    self.mj = mujoco
    self.m = mujoco.MjModel.from_xml_path(str(xml))
    self.d = mujoco.MjData(self.m)
    self.qadr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS]
    rng = np.random.default_rng(seed)
    m = self.m
    geoms, pts_local, areas = [], [], []
    for g in range(m.ngeom):
      if m.geom_group[g] > 2 or m.geom_rgba[g, 3] == 0:
        continue
      if bodies is not None and m.body(m.geom_bodyid[g]).name not in bodies:
        continue
      t = m.geom_type[g]
      if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        V = m.mesh_vert[v0:v0 + nv].astype(np.float64)
        F = m.mesh_face[f0:f0 + nf]
        tri = V[F]
        a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
      elif t in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
        s = m.geom_size[g].copy()
        if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
          s = np.array([s[0], s[0], s[1]])
        c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s
        quads = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
        tri = np.array([[c[q[0]], c[q[1]], c[q[2]]] for q in quads] + [[c[q[0]], c[q[2]], c[q[3]]] for q in quads])
        a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
      else:
        continue
      geoms.append(g); pts_local.append(tri); areas.append(a)
    tot = sum(a.sum() for a in areas)
    self.geom_ids, self.local, self.owner = [], [], []
    for g, tri, a in zip(geoms, pts_local, areas):
      n = max(50, int(round(n_points * a.sum() / tot)))
      k = rng.choice(len(tri), n, p=a / a.sum())
      r1, r2 = rng.random(n), rng.random(n)
      s1 = np.sqrt(r1)
      p = (1 - s1)[:, None] * tri[k, 0] + (s1 * (1 - r2))[:, None] * tri[k, 1] + (s1 * r2)[:, None] * tri[k, 2]
      self.local.append(np.vstack([p, tri.reshape(-1, 3)[:: max(1, len(tri) * 3 // 200)]]))
      self.geom_ids.append(g)
    self.area_m2 = tot

  def world_points(self, q_sim) -> np.ndarray:
    self.d.qpos[self.qadr] = q_sim
    self.mj.mj_kinematics(self.m, self.d)
    out = [self.d.geom_xpos[g] + p @ self.d.geom_xmat[g].reshape(3, 3).T for g, p in zip(self.geom_ids, self.local)]
    return np.vstack(out)

  def mask(self, p, q_sim=None, w: int = 160, h: int = 120, pts: np.ndarray | None = None, close_px: int = 3,
           pose=None) -> np.ndarray:
    """p: 7 camera params (only p[6] = fovy is used when `pose` = (pos, R) is given)."""
    import cv2
    if pts is None:
      pts = self.world_points(q_sim)
    pos, R = cam_pose(p) if pose is None else pose
    c = (pts - pos) @ R
    front = c[:, 2] < -1e-3
    c = c[front]
    f = 0.5 * h / np.tan(np.deg2rad(p[6]) / 2)
    u = (w / 2 + f * c[:, 0] / -c[:, 2]).astype(np.int32)
    v = (h / 2 - f * c[:, 1] / -c[:, 2]).astype(np.int32)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    m = np.zeros((h, w), np.uint8)
    m[v[ok], u[ok]] = 1
    if close_px > 1:
      m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    return m.astype(bool)
