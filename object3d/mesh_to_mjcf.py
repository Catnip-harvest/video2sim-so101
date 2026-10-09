"""Turn any triangle mesh (OBJ / GLB / PLY / STL) into a MuJoCo object the simulator can grasp.

    python -I object3d/mesh_to_mjcf.py MESH --out DIR --mass 0.094 --target-size 0.057 --target-dim x

What it writes into DIR:
  object.xml        MJCF: one free body "object" (joint "object_joint") with
                      * one VISUAL geom: the original (textured) mesh, contype=0 conaffinity=0,
                        group 1, mass 0. It is only drawn, never collides, never weighs anything;
                      * N COLLISION geoms: convex pieces of a repaired copy of the mesh, group 3,
                        carrying all of the mass (split by piece volume, or by area for "shell").
  visual.obj (+ texture.png when the source had one)
  collision_NNN.obj
  report.json       scale factor, bounding box before/after, piece count, watertight flag,
                    volumes, inertia mode, backend used, timings

Why the split:
  * MuJoCo collides every mesh as its CONVEX HULL. A mug as one mesh would be a solid lump
    with no hole, so the collision copy is cut into convex pieces first (convex decomposition).
  * The textured scan is usually not closed (holes, flipped faces), so its volume and inertia
    are undefined. The visual mesh therefore carries no mass; a repaired, closed copy feeds the
    decomposition, and the pieces carry the mass.

Steps (each is a function below, in this order):
  1. load            any trimesh-readable file; GLB/glTF is Y-up, so it is turned Z-up (--up)
  2. align           optional turn about the vertical so the footprint's smallest rectangle is
                     axis-aligned (the grasp planner squeezes along the body x or y axis)
  3. scale           one measured dimension in metres (--target-size + --target-dim) or an
                     explicit factor (--scale). Written as MJCF <mesh scale="s s s">, so the
                     files stay in source units and a rescale is one attribute.
  4. centre          body origin = centre of the bounding box (so "half height" is exact)
  5. repair          merge duplicate vertices, drop degenerate faces, fix winding, cap holes
                     with a fan; record whether the result is watertight (closed)
  6. decompose       CoACD (MIT) if it loads; otherwise V-HACD 4 via `vhacdx` (MIT bindings,
                     BSD-3 library); otherwise one convex hull. Every result must hold >= 90 %
                     of the closed mesh's volume or the next backend is tried. The backend
                     used, and why others were skipped, is recorded. (Laptop, 9 Oct 2026:
                     Windows Application Control blocked CoACD's DLL on the first import and
                     allowed it afterwards; vhacdx loads but returns broken pieces, e.g. 18-21 %
                     of the volume, so the coverage check rejects it.)
  7. mass/inertia    --inertia convex (solid objects) | shell (thin / hollow, mass on the
                     surface) | auto (convex if the repaired mesh is closed, else shell). Uses
                     MuJoCo's own <mesh inertia="..."> modes.
  8. write + check   the MJCF is compiled with MuJoCo; mass and piece count are asserted.

Licences of what this file imports: trimesh (MIT), numpy/scipy (BSD), mujoco (Apache-2.0),
coacd (MIT, optional), vhacdx (MIT; V-HACD is BSD-3, optional). No GPL code (no PyMeshLab, no
CGAL).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_FRICTION = (0.8, 0.005, 0.0001)  # same as the cube (task/so101_pickplace/mj_scene.py)
WORLD_CONTYPE = 1  # world things collide with the robot and each other (robot_model.py)
WORLD_CONAFFINITY = 1
MIN_PIECE_VOLUME_FRACTION = 1e-4  # pieces smaller than this share of the total are dropped
COLLISION_RGBA = (0.9, 0.6, 0.2, 0.5)


# --------------------------------------------------------------------------------------
# 1. load
# --------------------------------------------------------------------------------------
@dataclass
class Loaded:
  vertices: np.ndarray  # (n, 3) float64, source units, after the up-axis turn
  faces: np.ndarray  # (m, 3) int
  uv: np.ndarray | None  # (n, 2) per-vertex texture coordinates, if any
  texture_file: Path | None  # existing image file on disk (OBJ+MTL), if any
  texture_image: object | None  # PIL image embedded in the file (GLB), if any
  source: Path


def load_mesh(path: str | Path, up: str = "auto") -> Loaded:
  import trimesh

  path = Path(path)
  obj = trimesh.load(str(path), process=False)
  if isinstance(obj, trimesh.Scene):
    geoms = list(obj.dump()) if hasattr(obj, "dump") else list(obj.geometry.values())
    geoms = [g for g in geoms if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not geoms:
      raise ValueError(f"{path}: no triangle geometry")
    mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
  elif isinstance(obj, trimesh.Trimesh):
    mesh = obj
  else:
    raise ValueError(f"{path}: not a triangle mesh ({type(obj).__name__})")
  v = np.asarray(mesh.vertices, dtype=np.float64).copy()
  f = np.asarray(mesh.faces, dtype=np.int64).copy()
  uv, tex_file, tex_img = None, None, None
  vis = getattr(mesh, "visual", None)
  if vis is not None and getattr(vis, "kind", None) == "texture" and getattr(vis, "uv", None) is not None:
    if len(vis.uv) == len(v):
      uv = np.asarray(vis.uv, dtype=np.float64).copy()
      tex_file = _mtl_texture(path)
      if tex_file is None:
        tex_img = getattr(getattr(vis, "material", None), "image", None)
        if tex_img is None:
          tex_img = getattr(getattr(vis, "material", None), "baseColorTexture", None)
  if up == "auto":
    up = "y" if path.suffix.lower() in (".glb", ".gltf") else "z"
  if up == "y":  # glTF convention: +Y up, so turn +Y onto +Z (x stays, y -> z, z -> -y)
    v = v @ np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
  elif up != "z":
    raise ValueError("up must be auto|y|z")
  return Loaded(v, f, uv, tex_file, tex_img, path)


def _mtl_texture(obj_path: Path) -> Path | None:
  """map_Kd image named in the OBJ's material file, if it exists next to it."""
  if obj_path.suffix.lower() != ".obj":
    return None
  for line in obj_path.read_text(errors="ignore").splitlines()[:200]:
    if line.startswith("mtllib"):
      mtl = obj_path.parent / line.split(None, 1)[1].strip()
      if not mtl.is_file():
        return None
      for ml in mtl.read_text(errors="ignore").splitlines():
        if ml.strip().startswith("map_Kd"):
          img = obj_path.parent / ml.strip().split(None, 1)[1].strip()
          return img if img.is_file() else None
  return None


# --------------------------------------------------------------------------------------
# 2-4. align, scale, centre
# --------------------------------------------------------------------------------------
def footprint_yaw(vertices: np.ndarray) -> float:
  """Turn (rad, about +z) that makes the smallest-area rectangle around the top-down
  footprint axis-aligned, with its longer side on x."""
  from scipy.spatial import ConvexHull

  xy = vertices[:, :2]
  hull = xy[ConvexHull(xy).vertices]
  best = (math.inf, 0.0)
  for i in range(len(hull)):
    e = hull[(i + 1) % len(hull)] - hull[i]
    a = math.atan2(e[1], e[0])
    c, s = math.cos(-a), math.sin(-a)
    r = hull @ np.array([[c, s], [-s, c]])
    ext = r.max(0) - r.min(0)
    area = float(ext[0] * ext[1])
    if area < best[0] - 1e-12:
      best = (area, -a if ext[0] >= ext[1] else -a + math.pi / 2)
  yaw = (best[1] + math.pi / 2) % math.pi - math.pi / 2  # in [-90, 90) deg
  return float(yaw)


def rot_z(a: float) -> np.ndarray:
  c, s = math.cos(a), math.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def scale_factor(extents_src: np.ndarray, target_size: float | None, target_dim: str,
                 scale: float | None) -> float:
  if scale is not None:
    if target_size is not None:
      raise ValueError("give --scale OR --target-size, not both")
    return float(scale)
  if target_size is None:
    raise ValueError("a scale source is required: --target-size (m) with --target-dim, or --scale")
  idx = {"x": 0, "y": 1, "z": 2, "height": 2}
  if target_dim in idx:
    src = float(extents_src[idx[target_dim]])
  elif target_dim == "max":
    src = float(extents_src.max())
  elif target_dim == "min":
    src = float(extents_src.min())
  else:
    raise ValueError("target_dim must be x|y|z|height|max|min")
  if src <= 0:
    raise ValueError("mesh has zero size along the target dimension")
  return float(target_size) / src


# --------------------------------------------------------------------------------------
# 5. repair
# --------------------------------------------------------------------------------------
def repair(vertices: np.ndarray, faces: np.ndarray, merge_tol: float) -> tuple[object, dict]:
  """Closed (if possible) geometry-only copy for collision and mass. Returns
  (trimesh.Trimesh, info). Never touches the visual mesh."""
  import trimesh

  info: dict = {}
  info["input_watertight"] = bool(
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).is_watertight)
  # Merging near-duplicate vertices closes texture seams, but too coarse a merge can
  # collapse a real tiny triangle and leave a non-manifold edge. Try the requested
  # tolerance, then 100x finer, then exact duplicates only; keep the first closed result
  # (else the one with the fewest open edges). Last resort: exact merge keeping
  # zero-area faces (dropping one can open a pinhole next to a non-manifold edge).
  digits = max(1, int(round(-math.log10(merge_tol))))
  best = None
  for label, dg, drop_degenerate in (("tol", digits, True), ("tol/100", digits + 2, True),
                                     ("exact", None, True),
                                     ("exact_keep_degenerate", None, False)):
    m = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)
    if dg is None:
      _, idx, inv = np.unique(m.vertices, axis=0, return_index=True, return_inverse=True)
      m = trimesh.Trimesh(vertices=m.vertices[idx], faces=inv.ravel()[m.faces], process=False)
    else:
      m.merge_vertices(digits_vertex=dg, merge_tex=True, merge_norm=True)
    if drop_degenerate:
      m.update_faces(m.nondegenerate_faces())
      m.update_faces(_drop_fins(np.asarray(m.faces)))
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(m)
    holes_capped = 0
    if not m.is_watertight:
      m, holes_capped = cap_holes(m)
    open_edges = _open_edge_count(m)
    if best is None or open_edges < best[0]:
      best = (open_edges, m, holes_capped, label)
    if m.is_watertight:
      break
  _, m, holes_capped, label = best
  trimesh.repair.fix_inversion(m)
  info["merge"] = label
  info["boundary_loops_capped"] = holes_capped
  info["watertight"] = bool(m.is_watertight)
  info["is_volume"] = bool(m.is_volume)
  info["n_vertices"] = int(len(m.vertices))
  info["n_faces"] = int(len(m.faces))
  return m, info


def _drop_fins(f: np.ndarray) -> np.ndarray:
  """Mask that removes BOTH faces of a 'fin': the same three vertices listed twice with
  opposite winding (a zero-thickness double wall; scans produce them where two vertices
  coincide). Keeping one would leave a non-manifold edge."""
  key = np.sort(f, axis=1)
  _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
  inv = inv.ravel()
  keep = np.ones(len(f), bool)
  for g in np.nonzero(cnt == 2)[0]:
    i, j = np.nonzero(inv == g)[0]
    # same cyclic order = duplicate (handled by unique_faces); opposite = fin
    a = list(f[i])
    same = any(a[k:] + a[:k] == list(f[j]) for k in range(3))
    if not same:
      keep[i] = keep[j] = False
  return keep


def _open_edge_count(m) -> int:
  """Edges not shared by exactly two faces (0 for a closed, manifold surface)."""
  f = np.asarray(m.faces)
  e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
  _, cnt = np.unique(e, axis=0, return_counts=True)
  return int(np.sum(cnt != 2))


def cap_holes(m) -> tuple[object, int]:
  """Close every open boundary with a fan of triangles to one new centre vertex per
  connected boundary. Each boundary edge (a->b, as it runs in its only face) gets the
  triangle (b, a, centre), which keeps the winding consistent and pairs every new edge."""
  import trimesh

  f = np.asarray(m.faces)
  e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
  key = np.sort(e, axis=1)
  _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
  boundary = e[cnt[inv.ravel()] == 1]
  if len(boundary) == 0:
    return m, 0
  # connected components of the boundary graph (union-find on vertex ids)
  parent: dict[int, int] = {}

  def find(x):
    while parent.setdefault(x, x) != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  for a, b in boundary:
    ra, rb = find(int(a)), find(int(b))
    if ra != rb:
      parent[ra] = rb
  comp = np.array([find(int(a)) for a in boundary[:, 0]])
  verts = np.asarray(m.vertices)
  new_v, new_f = [verts], [f]
  n = len(verts)
  roots = np.unique(comp)
  for k, root in enumerate(roots):
    edges = boundary[comp == root]
    centre = verts[np.unique(edges.ravel())].mean(0)
    c = n + k
    new_v.append(centre[None])
    new_f.append(np.stack([edges[:, 1], edges[:, 0], np.full(len(edges), c)], axis=1))
  out = trimesh.Trimesh(vertices=np.concatenate(new_v), faces=np.concatenate(new_f), process=False)
  return out, int(len(roots))


# --------------------------------------------------------------------------------------
# 6. convex decomposition
# --------------------------------------------------------------------------------------
def available_backends() -> dict[str, str]:
  """backend -> 'ok' or the import error (so a blocked DLL is reported, not hidden)."""
  out = {}
  try:
    import coacd  # noqa: F401
    out["coacd"] = "ok"
  except Exception as e:  # noqa: BLE001  (Application Control raises OSError)
    out["coacd"] = f"{type(e).__name__}: {e}"
  try:
    import vhacdx  # noqa: F401
    out["vhacd"] = "ok"
  except Exception as e:  # noqa: BLE001
    out["vhacd"] = f"{type(e).__name__}: {e}"
  out["hull"] = "ok"
  return out


MIN_COVERAGE = 0.9  # pieces must hold >= 90 % of a closed mesh's volume, else reject them


def decompose(m, backend: str = "auto", threshold: float = 0.10, max_hulls: int = -1,
              max_hull_vertices: int = 64, vhacd_resolution: int = 400_000,
              seed: int = 0) -> tuple[list[np.ndarray], dict]:
  """Convex pieces of mesh m (vertex arrays; each piece is re-hulled with qhull so it is
  convex for sure). Returns (pieces, info).

  Coverage check: when m is closed, the pieces' total volume must be at least
  MIN_COVERAGE of m's volume (convex pieces over-cover concave spots, so > 1 is normal;
  well under 1 means pieces are missing). With backend="auto" a backend that fails the
  check is skipped and the next is tried; the reason is recorded. This is not
  hypothetical: vhacdx 0.1.0 on Windows returned a tetrahedron for a box and 39 % of a
  sphere's volume (measured 9 Oct 2026), so its output is never trusted unchecked."""
  avail = available_backends()
  order = ["coacd", "vhacd", "hull"] if backend == "auto" else [backend]
  info: dict = {"requested": backend, "available": avail, "rejected": {}}
  v = np.asarray(m.vertices, float)
  f = np.asarray(m.faces, np.int64)
  closed_volume = float(m.volume) if m.is_watertight and m.volume > 0 else None
  result = None
  for b in order:
    if avail.get(b) != "ok":
      continue
    t0 = time.perf_counter()
    if b == "coacd":
      import coacd

      coacd.set_log_level("error")
      parts = coacd.run_coacd(coacd.Mesh(v, f), threshold=threshold,
                              max_convex_hull=max_hulls if max_hulls > 0 else -1,
                              max_ch_vertex=max_hull_vertices, seed=seed)
      raw = [np.asarray(p[0], float) for p in parts]
      params = {"threshold": threshold, "max_convex_hull": max_hulls,
                "max_ch_vertex": max_hull_vertices, "seed": seed}
    elif b == "vhacd":
      import vhacdx

      parts = vhacdx.compute_vhacd(v, f.astype(np.uint32),
                                   maxConvexHulls=max_hulls if max_hulls > 0 else 32,
                                   resolution=vhacd_resolution,
                                   maxNumVerticesPerCH=max_hull_vertices,
                                   minimumVolumePercentErrorAllowed=1.0, asyncACD=False)
      raw = [np.asarray(p[0], float) for p in parts]
      params = {"maxConvexHulls": max_hulls if max_hulls > 0 else 32,
                "resolution": vhacd_resolution, "maxNumVerticesPerCH": max_hull_vertices}
    else:
      raw = [v]
      params = {}
    pieces, vols = [], []
    for p in raw:
      hv, vol = _hull(p)
      if hv is not None:
        pieces.append(hv)
        vols.append(vol)
    total = sum(vols)
    keep = [i for i, vol in enumerate(vols) if vol >= MIN_PIECE_VOLUME_FRACTION * total]
    coverage = (sum(vols[i] for i in keep) / closed_volume) if closed_volume else None
    secs = time.perf_counter() - t0
    if coverage is not None and coverage < MIN_COVERAGE and backend == "auto" and b != "hull":
      info["rejected"][b] = (f"pieces hold only {coverage:.0%} of the mesh volume "
                             f"({len(keep)} pieces, {secs:.1f} s)")
      continue
    result = ([pieces[i] for i in keep], b, params, len(pieces) - len(keep), coverage, secs)
    break
  if result is None:
    raise RuntimeError(f"no convex-decomposition backend could run: {avail}, {info['rejected']}")
  pieces, b, params, dropped, coverage, secs = result
  info.update(backend=b, params=params, pieces_dropped_tiny=dropped,
              coverage_of_mesh_volume=coverage, seconds=round(secs, 2))
  if coverage is not None and coverage < MIN_COVERAGE:
    info["warning"] = f"pieces hold only {coverage:.0%} of the mesh volume"
  return pieces, info


def _hull(points: np.ndarray) -> tuple[np.ndarray | None, float]:
  from scipy.spatial import ConvexHull, QhullError

  if len(points) < 4:
    return None, 0.0
  try:
    h = ConvexHull(points)
  except QhullError:
    return None, 0.0
  return points[h.vertices], float(h.volume)


def hull_faces(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Outward-wound triangles of the convex hull of points."""
  from scipy.spatial import ConvexHull

  h = ConvexHull(points)
  tri = h.simplices.copy()
  c = points.mean(0)
  for i, t in enumerate(tri):
    n = np.cross(points[t[1]] - points[t[0]], points[t[2]] - points[t[0]])
    if np.dot(n, points[t[0]] - c) < 0:
      tri[i] = t[[0, 2, 1]]
  used = np.unique(tri)
  remap = -np.ones(len(points), int)
  remap[used] = np.arange(len(used))
  return points[used], remap[tri]


def hull_area(points: np.ndarray) -> float:
  from scipy.spatial import ConvexHull

  return float(ConvexHull(points).area)


# --------------------------------------------------------------------------------------
# 7-8. write
# --------------------------------------------------------------------------------------
def write_obj(path: Path, v: np.ndarray, f: np.ndarray, uv: np.ndarray | None = None) -> None:
  lines = [f"# written by object3d/mesh_to_mjcf.py: {len(v)} vertices, {len(f)} faces"]
  lines += [f"v {a:.9g} {b:.9g} {c:.9g}" for a, b, c in v]
  if uv is not None:
    lines += [f"vt {a:.7g} {b:.7g}" for a, b in uv]
    lines += [f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}" for a, b, c in f]
  else:
    lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in f]
  path.write_text("\n".join(lines) + "\n")


def _fmt(x) -> str:
  return " ".join(f"{float(a):.9g}" for a in np.atleast_1d(x))


def build_mjcf(name: str, scale: float, n_pieces: int, piece_masses: list[float],
               inertia_mode: str, friction, has_texture: bool, rgba=(0.8, 0.8, 0.8, 1.0)) -> str:
  root = ET.Element("mujoco", model=name)
  ET.SubElement(root, "compiler", angle="radian")
  asset = ET.SubElement(root, "asset")
  s = _fmt([scale] * 3)
  if has_texture:
    ET.SubElement(asset, "texture", name=f"{name}_tex", type="2d", file="texture.png")
    ET.SubElement(asset, "material", name=f"{name}_mat", texture=f"{name}_tex", specular="0.2")
  else:
    ET.SubElement(asset, "material", name=f"{name}_mat", rgba=_fmt(rgba))
  # The visual mesh may be open; "shell" inertia never needs a closed volume (it weighs 0).
  ET.SubElement(asset, "mesh", name=f"{name}_visual", file="visual.obj", scale=s, inertia="shell")
  for i in range(n_pieces):
    ET.SubElement(asset, "mesh", name=f"{name}_col_{i:03d}", file=f"collision_{i:03d}.obj",
                  scale=s, inertia=inertia_mode)
  wb = ET.SubElement(root, "worldbody")
  body = ET.SubElement(wb, "body", name="object")
  ET.SubElement(body, "freejoint", name="object_joint")
  ET.SubElement(body, "geom", name="object_visual", type="mesh", mesh=f"{name}_visual",
                material=f"{name}_mat", contype="0", conaffinity="0", group="1", mass="0")
  for i, mass in enumerate(piece_masses):
    ET.SubElement(body, "geom", name=f"object_col_{i:03d}", type="mesh",
                  mesh=f"{name}_col_{i:03d}", mass=f"{mass:.9g}", friction=_fmt(friction),
                  condim="4", contype=str(WORLD_CONTYPE), conaffinity=str(WORLD_CONAFFINITY),
                  group="3", rgba=_fmt(COLLISION_RGBA))
  ET.indent(root, space="  ")
  return ET.tostring(root, encoding="unicode") + "\n"


def convert(mesh_path: str | Path, out_dir: str | Path, mass: float,
            target_size: float | None = None, target_dim: str = "max",
            scale: float | None = None, name: str | None = None, up: str = "auto",
            align: bool = True, inertia: str = "auto", backend: str = "auto",
            threshold: float = 0.10, max_hulls: int = -1, max_hull_vertices: int = 64,
            vhacd_resolution: int = 400_000, friction=DEFAULT_FRICTION, seed: int = 0) -> dict:
  """Run steps 1-8. Returns the report dict (also written to out_dir/report.json)."""
  import mujoco

  if mass <= 0:
    raise ValueError("mass must be > 0 kg")
  if inertia not in ("auto", "convex", "shell"):
    raise ValueError("inertia must be auto|convex|shell")
  t0 = time.perf_counter()
  timing = {}
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  for old in out.glob("collision_*.obj"):
    old.unlink()
  src = load_mesh(mesh_path, up=up)
  name = name or Path(mesh_path).stem
  name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
  bbox_src = np.stack([src.vertices.min(0), src.vertices.max(0)])
  timing["load_s"] = time.perf_counter() - t0

  # 2. align: turn about +z so the footprint's minimum rectangle is axis-aligned
  yaw = footprint_yaw(src.vertices) if align else 0.0
  v = src.vertices @ rot_z(yaw).T
  # 3. scale (from the aligned extents) and 4. centre on the bounding box
  ext_src = v.max(0) - v.min(0)
  s = scale_factor(ext_src, target_size, target_dim, scale)
  centre = (v.max(0) + v.min(0)) / 2
  v = v - centre
  bbox_m = np.stack([v.min(0), v.max(0)]) * s
  t1 = time.perf_counter()

  # 5. repair (merge tolerance: 1e-5 of the object size, in source units; textured scans
  # often store seam vertices with ~6 significant digits)
  rep, rep_info = repair(v, src.faces, merge_tol=max(float(ext_src.max()) * 1e-5, 1e-12))
  timing["repair_s"] = time.perf_counter() - t1
  watertight = rep_info["watertight"]
  mode = inertia if inertia != "auto" else ("convex" if watertight else "shell")

  # 6. decompose (in metres, so thresholds mean the same for every source unit)
  t2 = time.perf_counter()
  rep_m = rep.copy()
  rep_m.vertices = np.asarray(rep.vertices) * s
  pieces_m, dec_info = decompose(rep_m, backend=backend, threshold=threshold, max_hulls=max_hulls,
                                 max_hull_vertices=max_hull_vertices,
                                 vhacd_resolution=vhacd_resolution, seed=seed)
  timing["decompose_s"] = time.perf_counter() - t2

  # 7. mass split: by piece volume (solid) or piece surface area (shell)
  vols = np.array([_hull(p)[1] for p in pieces_m])
  areas = np.array([hull_area(p) for p in pieces_m])
  w = vols if mode == "convex" else areas
  piece_masses = (mass * w / w.sum()).tolist()

  # 8. write files (meshes in source units; MJCF scale does the metres)
  has_tex = src.uv is not None and (src.texture_file is not None or src.texture_image is not None)
  write_obj(out / "visual.obj", v, src.faces, src.uv if has_tex else None)
  if has_tex:
    if src.texture_file is not None:
      if src.texture_file.suffix.lower() == ".png":
        shutil.copyfile(src.texture_file, out / "texture.png")
      else:
        from PIL import Image

        Image.open(src.texture_file).save(out / "texture.png")
    else:
      src.texture_image.save(out / "texture.png")
  for i, p in enumerate(pieces_m):
    hv, hf = hull_faces(p / s)
    write_obj(out / f"collision_{i:03d}.obj", hv, hf)
  xml = build_mjcf(name, s, len(pieces_m), piece_masses, mode, friction, has_tex)
  (out / "object.xml").write_text(xml)

  # compile check
  t3 = time.perf_counter()
  model = mujoco.MjModel.from_xml_path(str(out / "object.xml"))
  timing["compile_s"] = time.perf_counter() - t3
  b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
  assert abs(model.body_mass[b] - mass) < 1e-6 * max(1.0, mass), (model.body_mass[b], mass)
  ncol = int(np.sum(model.geom_contype != 0))
  assert ncol == len(pieces_m)
  timing["total_s"] = time.perf_counter() - t0

  mesh_volume = float(rep_m.volume) if watertight else None
  hull_sum = float(vols.sum())
  report = {
    "name": name,
    "source": str(Path(mesh_path).resolve()),
    "up_axis": up if up != "auto" else ("y" if src.source.suffix.lower() in (".glb", ".gltf") else "z"),
    "yaw_align_deg": round(math.degrees(yaw), 3),
    "scale_source": ({"explicit_factor": scale} if scale is not None
                     else {"target_size_m": target_size, "target_dim": target_dim}),
    "scale_factor": s,
    "bbox_source_units": bbox_src.tolist(),
    "extents_source_units_after_align": ext_src.tolist(),
    "bbox_m": bbox_m.tolist(),
    "extents_m": (bbox_m[1] - bbox_m[0]).tolist(),
    "half_extents_m": ((bbox_m[1] - bbox_m[0]) / 2).tolist(),
    "body_origin": "centre of the axis-aligned bounding box (after the yaw turn)",
    "mass_kg": mass,
    "inertia_mode": mode,
    "inertia_requested": inertia,
    "repair": rep_info,
    "watertight": watertight,
    "mesh_volume_m3": mesh_volume,
    "hull_volume_sum_m3": hull_sum,
    "hull_to_mesh_volume_ratio": (hull_sum / mesh_volume) if mesh_volume else None,
    "density_kg_m3_from_mesh_volume": (mass / mesh_volume) if mesh_volume else None,
    "n_hulls": len(pieces_m),
    "hull_vertex_counts": [int(len(p)) for p in pieces_m],
    "decomposition": dec_info,
    "visual": {"vertices": int(len(v)), "faces": int(len(src.faces)), "textured": bool(has_tex)},
    "friction": list(friction),
    "compiled": {
      "body_mass_kg": float(model.body_mass[b]),
      "body_inertia_diag_kg_m2": model.body_inertia[b].tolist(),
      "com_in_body_m": model.body_ipos[b].tolist(),
    },
    "timing_s": {k: round(val, 3) for k, val in timing.items()},
  }
  (out / "report.json").write_text(json.dumps(report, indent=2))
  return report


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("mesh", help="input mesh: .obj .glb .gltf .ply .stl")
  ap.add_argument("--out", required=True, help="output folder (object.xml, meshes, report.json)")
  ap.add_argument("--mass", type=float, required=True, help="object mass in kg (kitchen scale)")
  g = ap.add_argument_group("scale source (one of)")
  g.add_argument("--target-size", type=float, help="measured size in metres of --target-dim")
  g.add_argument("--target-dim", default="max", choices=["x", "y", "z", "height", "max", "min"],
                 help="which aligned bounding-box side --target-size measures (default max)")
  g.add_argument("--scale", type=float, help="explicit factor: source units -> metres")
  ap.add_argument("--name", help="asset name prefix (default: file stem)")
  ap.add_argument("--up", default="auto", choices=["auto", "y", "z"],
                  help="source up axis (auto: y for glTF/GLB, else z)")
  ap.add_argument("--no-align", action="store_true", help="skip the footprint yaw alignment")
  ap.add_argument("--inertia", default="auto", choices=["auto", "convex", "shell"],
                  help="convex = solid, shell = thin/hollow; auto = convex if closed else shell")
  ap.add_argument("--backend", default="auto", choices=["auto", "coacd", "vhacd", "hull"])
  ap.add_argument("--threshold", type=float, default=0.10,
                  help="CoACD concavity threshold (0.10: 9-32 pieces, 40-270 s on the YCB scans)")
  ap.add_argument("--max-hulls", type=int, default=-1,
                  help="cap on pieces (-1 = none; a cap makes CoACD merge pieces, which is slow)")
  ap.add_argument("--max-hull-vertices", type=int, default=64)
  ap.add_argument("--vhacd-resolution", type=int, default=400_000)
  ap.add_argument("--friction", type=float, nargs=3, default=list(DEFAULT_FRICTION))
  a = ap.parse_args(argv)
  rep = convert(a.mesh, a.out, a.mass, target_size=a.target_size, target_dim=a.target_dim,
                scale=a.scale, name=a.name, up=a.up, align=not a.no_align, inertia=a.inertia,
                backend=a.backend, threshold=a.threshold, max_hulls=a.max_hulls,
                max_hull_vertices=a.max_hull_vertices, vhacd_resolution=a.vhacd_resolution,
                friction=tuple(a.friction))
  short = {k: rep[k] for k in ("name", "scale_factor", "extents_m", "watertight", "n_hulls",
                               "inertia_mode", "mesh_volume_m3", "hull_to_mesh_volume_ratio")}
  short["backend"] = rep["decomposition"]["backend"]
  short["timing_s"] = rep["timing_s"]
  print(json.dumps(short, indent=2))


if __name__ == "__main__":
  main()
