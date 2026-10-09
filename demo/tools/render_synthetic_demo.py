"""Render a synthetic "watch me once" clip with exact ground truth.

    python -I tools/render_synthetic_demo.py --out test_assets/synthetic_pick_place

Scene: a plain wooden table, an A4 sheet lying flat, a 40 mm red cube, a blue
bowl, and a capsule-built skin-coloured right hand that approaches, pinches the
cube between thumb and index, lifts it, carries it over the bowl, lowers it in,
opens, and retreats. The camera is a fixed "phone" in a 3/4 view from the side,
~0.35 m above the table, as the filming guide asks.

Rendering is a small NumPy ray caster (no OpenGL: the agent shell on this laptop
has no WGL context, so MuJoCo's renderer cannot start). It gives exact ground
truth for free: per-frame cube masks, 3D keypoints, camera intrinsics/extrinsics.

Outputs:
  <out>.mp4          the clip (mp4v, 30 fps, 960x540)
  <out>_gt.json      camera, A4 corners, cube pose per frame, 21 hand keypoints
                     per frame (MediaPipe order), grasp/release times
  <out>_gt_masks.npz packed boolean cube masks per frame

This exercises the full extractor, including MediaPipe on a rendered hand. It
does NOT validate real-phone-video quality.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------- scene setup
WIDTH, HEIGHT, FPS, DURATION_S = 960, 540, 30, 7.0
FOVY_DEG = 42.0
CAMERA_POS = np.array([0.42, -0.36, 0.31])
CAMERA_TARGET = np.array([0.0, 0.10, 0.05])

A4_CENTER = np.array([0.0, -0.02])
A4_YAW = np.deg2rad(14.0)
A4_SIZE = (0.297, 0.210)

CUBE_SIZE = 0.040
CUBE_START_XY = np.array([-0.13, 0.12])

BOWL_CENTER_XY = np.array([0.15, 0.15])
BOWL_RADIUS = 0.075           # spherical shell radius
BOWL_RIM_Z = 0.048
BOWL_FLOOR_Z = 0.008          # flat inner floor; the cube rests on it

TABLE_HALF = np.array([0.70, 0.50])
TABLE_CENTER = np.array([0.0, 0.05])

LIGHT_DIR = np.array([0.35, -0.25, 1.0]) / np.linalg.norm([0.35, -0.25, 1.0])

COLORS = {  # linear-ish RGB in [0, 1]
    "paper": np.array([0.95, 0.95, 0.94]),
    "cube": np.array([0.85, 0.16, 0.10]),
    "bowl": np.array([0.28, 0.52, 0.85]),
    "skin": np.array([0.90, 0.69, 0.57]),
    "sleeve": np.array([0.16, 0.21, 0.36]),
    "background": np.array([0.78, 0.78, 0.75]),
}


def look_at(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rows: camera x (right), y (down), z (forward) in world coords (OpenCV)."""
    forward = target - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


def make_table_texture() -> np.ndarray:
    """Plain light-wood table: low-contrast grain, like an ordinary desk."""
    rng = np.random.default_rng(3)
    h, w = 512, 512
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grain = np.sin(xx / 7.0 + 3.0 * np.sin(yy / 53.0)) * 0.5 + 0.5
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 3)
    shade = 0.84 + 0.05 * grain + 0.03 * noise
    base = np.array([0.78, 0.62, 0.45], np.float32)  # RGB light oak
    return np.clip(base[None, None, :] * shade[..., None], 0, 1)


# ------------------------------------------------------------- ray primitives
@dataclass
class Hit:
    t: np.ndarray       # (N,) distance, inf where no hit
    normal: np.ndarray  # (N, 3)


def _empty(n: int) -> Hit:
    return Hit(np.full(n, np.inf, np.float64), np.zeros((n, 3)))


def hit_capsule(o: np.ndarray, d: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> Hit:
    """Ray/capsule intersection (Inigo Quilez's formulation), vectorised."""
    n = len(d)
    out = _empty(n)
    ba = b - a
    oa = o - a
    baba = ba @ ba
    bard = d @ ba
    baoa = oa @ ba if oa.ndim == 1 else oa @ ba
    rdoa = np.einsum("ij,ij->i", d, np.broadcast_to(oa, d.shape))
    oaoa = (oa * oa).sum(-1) if oa.ndim > 1 else oa @ oa
    A = baba - bard * bard
    B = baba * rdoa - baoa * bard
    C = baba * oaoa - baoa * baoa - r * r * baba
    h = B * B - A * C
    ok = h >= 0
    with np.errstate(invalid="ignore", divide="ignore"):
        t_body = (-B - np.sqrt(np.where(ok, h, 0))) / A
        y = baoa + t_body * bard
        body = ok & (y > 0) & (y < baba) & (t_body > 1e-6)
        # end caps
        oc = np.where((y <= 0)[:, None], np.broadcast_to(oa, d.shape), np.broadcast_to(o - b, d.shape))
        b2 = np.einsum("ij,ij->i", d, oc)
        c2 = (oc * oc).sum(-1) - r * r
        h2 = b2 * b2 - c2
        t_cap = -b2 - np.sqrt(np.where(h2 > 0, h2, 0))
        cap = ok & ~body & (h2 > 0) & (t_cap > 1e-6)
    t = np.where(body, t_body, np.where(cap, t_cap, np.inf))
    hit = np.isfinite(t)
    if hit.any():
        p = (np.broadcast_to(o, d.shape)[hit] + t[hit, None] * d[hit])
        pa = p - a
        s = np.clip((pa @ ba) / baba, 0, 1)
        nrm = pa - s[:, None] * ba
        out.normal[hit] = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    out.t = t
    return out


def hit_ellipsoid(o, d, center, rot, radii) -> Hit:
    n = len(d)
    out = _empty(n)
    ol = (np.broadcast_to(o, d.shape) - center) @ rot / radii
    dl = (d @ rot) / radii
    A = (dl * dl).sum(-1)
    B = (ol * dl).sum(-1)
    C = (ol * ol).sum(-1) - 1
    h = B * B - A * C
    ok = h > 0
    with np.errstate(invalid="ignore"):
        t = (-B - np.sqrt(np.where(ok, h, 0))) / A
    t = np.where(ok & (t > 1e-6), t, np.inf)
    hit = np.isfinite(t)
    if hit.any():
        pl = ol[hit] + t[hit, None] * dl[hit]
        nrm = (pl / radii) @ rot.T
        out.normal[hit] = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    out.t = t
    return out


def hit_box(o, d, center, rot, half) -> Hit:
    n = len(d)
    out = _empty(n)
    ol = (np.broadcast_to(o, d.shape) - center) @ rot
    dl = d @ rot
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(dl) < 1e-12, 1e-12, dl)
        t1 = (-half - ol) * inv
        t2 = (half - ol) * inv
    tmin_axes = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2).min(-1)
    tmin = tmin_axes.max(-1)
    ok = (tmax >= np.maximum(tmin, 0)) & (tmin > 1e-6)
    t = np.where(ok, tmin, np.inf)
    hit = np.isfinite(t)
    if hit.any():
        axis = tmin_axes[hit].argmax(-1)
        nl = np.zeros((hit.sum(), 3))
        nl[np.arange(len(axis)), axis] = -np.sign(dl[hit][np.arange(len(axis)), axis])
        out.normal[hit] = nl @ rot.T
    out.t = t
    return out


def hit_bowl(o, d) -> Hit:
    """Open spherical bowl, two-sided, with a flat inner floor."""
    n = len(d)
    out = _empty(n)
    center = np.array([*BOWL_CENTER_XY, BOWL_RADIUS])
    oc = np.broadcast_to(o, d.shape) - center
    B = (oc * d).sum(-1)
    C = (oc * oc).sum(-1) - BOWL_RADIUS ** 2
    h = B * B - C
    ok = h > 0
    sq = np.sqrt(np.where(ok, h, 0))
    best = np.full(n, np.inf)
    for t_root in (-B - sq, -B + sq):
        z = np.broadcast_to(o, d.shape)[:, 2] + t_root * d[:, 2]
        valid = ok & (t_root > 1e-6) & (z <= BOWL_RIM_Z) & (z >= BOWL_FLOOR_Z)
        best = np.where(valid & (t_root < best), t_root, best)
    # flat floor disc
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = (BOWL_FLOOR_Z - np.broadcast_to(o, d.shape)[:, 2]) / d[:, 2]
    pf = np.broadcast_to(o, d.shape) + t_floor[:, None] * d
    floor_r2 = BOWL_RADIUS ** 2 - (BOWL_RADIUS - BOWL_FLOOR_Z) ** 2
    on_floor = (t_floor > 1e-6) & (((pf[:, :2] - BOWL_CENTER_XY) ** 2).sum(-1) <= floor_r2)
    use_floor = on_floor & (t_floor < best)
    best = np.where(use_floor, t_floor, best)
    hit = np.isfinite(best)
    if hit.any():
        p = np.broadcast_to(o, d.shape)[hit] + best[hit, None] * d[hit]
        nrm = (p - center) / BOWL_RADIUS
        nrm[use_floor[hit]] = [0, 0, 1]
        flip = (nrm * d[hit]).sum(-1) > 0          # two-sided: face the viewer
        nrm[flip] *= -1
        out.normal[hit] = nrm
    out.t = best
    return out


def hit_table(o, d) -> Hit:
    n = len(d)
    out = _empty(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -np.broadcast_to(o, d.shape)[:, 2] / d[:, 2]
    p = np.broadcast_to(o, d.shape) + t[:, None] * d
    inside = np.all(np.abs(p[:, :2] - TABLE_CENTER) <= TABLE_HALF, axis=-1)
    t = np.where((t > 1e-6) & inside, t, np.inf)
    out.t = t
    out.normal[np.isfinite(t)] = [0, 0, 1]
    return out


# ------------------------------------------------------------------ hand model
# Local hand frame: origin at wrist, +Y toward the fingers, +Z = palm normal
# (palm faces +Z), +X toward the thumb -> a right hand.
MCP = {
    "index": np.array([0.024, 0.088, 0.0]),
    "middle": np.array([0.002, 0.092, 0.0]),
    "ring": np.array([-0.018, 0.087, 0.0]),
    "pinky": np.array([-0.036, 0.077, 0.0]),
}
LENGTHS = {
    "index": (0.042, 0.025, 0.022),
    "middle": (0.046, 0.029, 0.024),
    "ring": (0.043, 0.027, 0.023),
    "pinky": (0.034, 0.020, 0.020),
}
ABDUCTION = {"index": 0.10, "middle": 0.0, "ring": -0.08, "pinky": -0.18}
FINGER_RADIUS = {"index": 0.0085, "middle": 0.0088, "ring": 0.0083, "pinky": 0.0074}
THUMB_CMC = np.array([0.020, 0.028, 0.012])
THUMB_LENGTHS = (0.042, 0.032, 0.028)
THUMB_RADIUS = 0.0098

FLEX_OPEN = {"index": (0.10, 0.10, 0.05), "middle": (0.15, 0.15, 0.10),
             "ring": (0.18, 0.18, 0.10), "pinky": (0.22, 0.20, 0.12)}
FLEX_CLOSED = {"index": (0.75, 0.55, 0.25), "middle": (0.95, 1.05, 0.55),
               "ring": (1.05, 1.10, 0.60), "pinky": (1.10, 1.10, 0.60)}
THUMB_OPEN = np.array([0.95, 0.45, 0.10, 0.10])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def finger_chain(name: str, flex) -> list[np.ndarray]:
    pts = [MCP[name]]
    total = 0.0
    for length, theta in zip(LENGTHS[name], flex):
        total += theta
        direction = rot_z(ABDUCTION[name]) @ np.array([0.0, np.cos(total), np.sin(total)])
        pts.append(pts[-1] + length * direction)
    return pts  # MCP, PIP, DIP, TIP


def thumb_chain(q: np.ndarray) -> list[np.ndarray]:
    a, b, f1, f2 = q
    d1 = np.array([np.sin(a) * np.cos(b), np.cos(a) * np.cos(b), np.sin(b)])
    flex_axis = -np.cross(d1, np.array([0.0, 0.0, 1.0]))
    flex_axis /= np.linalg.norm(flex_axis)
    d2 = axis_angle(flex_axis, f1) @ d1
    d3 = axis_angle(flex_axis, f1 + f2) @ d1
    mcp = THUMB_CMC + THUMB_LENGTHS[0] * d1
    ip = mcp + THUMB_LENGTHS[1] * d2
    tip = ip + THUMB_LENGTHS[2] * d3
    return [THUMB_CMC, mcp, ip, tip]


def solve_thumb_pinch() -> np.ndarray:
    index_tip = finger_chain("index", FLEX_CLOSED["index"])[-1]
    target = index_tip + np.array([0.004, -0.004, 0.006])

    def cost(q: np.ndarray) -> float:
        return float(np.sum((thumb_chain(q)[-1] - target) ** 2) * 1e4
                     + 0.01 * np.sum((q - THUMB_OPEN) ** 2))

    return minimize(cost, THUMB_OPEN.copy(), method="Nelder-Mead",
                    options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-10}).x


THUMB_PINCH = solve_thumb_pinch()


def hand_keypoints_local(closure: float) -> np.ndarray:
    """21 keypoints in MediaPipe order, local hand frame."""
    c = float(np.clip(closure, 0.0, 1.0))
    kp = [np.zeros(3)]
    kp += thumb_chain((1 - c) * THUMB_OPEN + c * THUMB_PINCH)
    for name in ("index", "middle", "ring", "pinky"):
        flex = tuple((1 - c) * o + c * k for o, k in zip(FLEX_OPEN[name], FLEX_CLOSED[name]))
        kp += finger_chain(name, flex)
    return np.array(kp)


def aperture(closure: float) -> float:
    kp = hand_keypoints_local(closure)
    return float(np.linalg.norm(kp[4] - kp[8]))


def hand_rotation(yaw_offset: float) -> np.ndarray:
    """Hand (local->world): fingers point forward (-y) and down, palm faces down."""
    fingers = np.array([0.20, -0.62, -0.76])
    fingers = rot_z(yaw_offset) @ (fingers / np.linalg.norm(fingers))
    up = np.array([0.0, 0.0, 1.0])
    palm = -(up - up.dot(fingers) * fingers)
    palm /= np.linalg.norm(palm)
    thumb_side = np.cross(fingers, palm)
    return np.column_stack([thumb_side, fingers, palm])


def hand_primitives(kp: np.ndarray, rot: np.ndarray, wrist: np.ndarray) -> list[tuple]:
    """Capsules/ellipsoid making up the rendered hand + forearm + sleeve."""
    fingers = rot[:, 1]
    prims = [
        ("capsule", wrist - 0.02 * fingers, wrist - 0.11 * fingers, 0.026, "skin"),
        ("capsule", wrist - 0.11 * fingers, wrist - 0.42 * fingers + np.array([0, 0, 0.10]), 0.036, "sleeve"),
        ("ellipsoid", wrist + rot @ np.array([-0.004, 0.048, 0.003]), rot, np.array([0.040, 0.050, 0.014]), "skin"),
    ]
    for mcp_index in (5, 9, 13, 17):
        prims.append(("capsule", kp[0], kp[mcp_index], 0.012, "skin"))
    for a, b in ((5, 9), (9, 13), (13, 17)):
        prims.append(("capsule", kp[a], kp[b], 0.011, "skin"))
    prims.append(("capsule", kp[0], kp[1], 0.016, "skin"))
    prims.append(("capsule", kp[1], kp[5], 0.011, "skin"))
    radii = {1: THUMB_RADIUS, 5: FINGER_RADIUS["index"], 9: FINGER_RADIUS["middle"],
             13: FINGER_RADIUS["ring"], 17: FINGER_RADIUS["pinky"]}
    for base, radius in radii.items():
        chain = [kp[base], kp[base + 1], kp[base + 2], kp[base + 3]]
        for j in range(3):
            a, b = chain[j], chain[j + 1]
            r = radius * (1.0 - 0.06 * j)
            if j == 2:  # rounded fingertip ends exactly at the TIP keypoint
                dvec = b - a
                b = b - r * dvec / max(np.linalg.norm(dvec), 1e-9)
            prims.append(("capsule", a, b, r, "skin"))
    return prims


# --------------------------------------------------------------- render frame
class RayCaster:
    def __init__(self) -> None:
        self.r_cv = look_at(CAMERA_POS, CAMERA_TARGET)
        self.t_cv = -self.r_cv @ CAMERA_POS
        fy = (HEIGHT / 2) / np.tan(np.deg2rad(FOVY_DEG) / 2)
        self.K = np.array([[fy, 0, WIDTH / 2], [0, fy, HEIGHT / 2], [0, 0, 1.0]])
        uu, vv = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
        rays_cam = np.stack([(uu - WIDTH / 2) / fy, (vv - HEIGHT / 2) / fy, np.ones_like(uu)], -1).reshape(-1, 3)
        rays_world = rays_cam @ self.r_cv
        self.dirs = (rays_world / np.linalg.norm(rays_world, axis=1, keepdims=True)).astype(np.float32)
        self.texture = make_table_texture()
        # static part of the scene (table, bowl) is the same every frame
        self.table = hit_table(CAMERA_POS, self.dirs)
        self.bowl = hit_bowl(CAMERA_POS, self.dirs)

    @staticmethod
    def bounding_sphere(prims) -> tuple[np.ndarray, float]:
        pts, pads = [], []
        for prim in prims:
            if prim[0] == "capsule":
                pts += [prim[1], prim[2]]
                pads += [prim[3], prim[3]]
            else:
                pts.append(prim[1])
                pads.append(float(np.max(prim[3])))
        pts = np.array(pts)
        center = pts.mean(0)
        radius = float(np.max(np.linalg.norm(pts - center, axis=1) + np.array(pads))) + 1e-3
        return center, radius

    @staticmethod
    def rays_near_sphere(o, d, center, radius) -> np.ndarray:
        oc = np.broadcast_to(o, d.shape) - center
        b = (oc * d).sum(-1)
        c = (oc * oc).sum(-1) - radius * radius
        return (b * b - c > 0) & ((-b + np.sqrt(np.maximum(b * b - c, 0))) > 0)

    def occluders(self, o, d, cube_pose, prims) -> np.ndarray:
        """True where a shadow ray from o along d hits the cube, bowl or hand."""
        blocked = np.isfinite(hit_box(o, d, cube_pose[0], cube_pose[1], np.full(3, CUBE_SIZE / 2)).t)
        blocked |= np.isfinite(hit_bowl(o, d).t)
        center, radius = self.bounding_sphere(prims)
        near = np.where(self.rays_near_sphere(o, d, center, radius))[0]
        if len(near):
            o_sub = o[near] if o.ndim > 1 else o
            for prim in prims:
                blocked[near] |= np.isfinite(self._hit_prim(o_sub, d[near], prim).t)
        return blocked

    @staticmethod
    def _hit_prim(o, d, prim) -> Hit:
        if prim[0] == "capsule":
            return hit_capsule(o, d, prim[1], prim[2], prim[3])
        return hit_ellipsoid(o, d, prim[1], prim[2], prim[3])

    def render(self, cube_pose, prims) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.dirs)
        best_t = self.table.t.copy()
        normal = self.table.normal.copy()
        label = np.where(np.isfinite(best_t), 1, 0)  # 0 bg, 1 table, 2 bowl, 3 cube, 4+ hand prims

        def take(hit: Hit, lab: int) -> None:
            nonlocal best_t, normal, label
            closer = hit.t < best_t
            best_t = np.where(closer, hit.t, best_t)
            normal[closer] = hit.normal[closer]
            label = np.where(closer, lab, label)

        take(self.bowl, 2)
        take(hit_box(CAMERA_POS, self.dirs, cube_pose[0], cube_pose[1], np.full(3, CUBE_SIZE / 2)), 3)
        center, radius = self.bounding_sphere(prims)
        near = np.where(self.rays_near_sphere(CAMERA_POS, self.dirs, center, radius))[0]
        for k, prim in enumerate(prims):
            sub = self._hit_prim(CAMERA_POS, self.dirs[near], prim)
            full = _empty(n)
            full.t[near] = sub.t
            full.normal[near] = sub.normal
            take(full, 4 + k)

        hit = np.isfinite(best_t)
        points = np.zeros((n, 3))
        points[hit] = CAMERA_POS + best_t[hit, None] * self.dirs[hit]

        albedo = np.tile(COLORS["background"], (n, 1))
        table_px = label == 1
        tex = self.texture
        uv = ((points[table_px, :2] + 1.0) * 3.0 % 1.0 * (tex.shape[0] - 1)).astype(int)
        albedo[table_px] = tex[uv[:, 1], uv[:, 0]]
        c, s = np.cos(A4_YAW), np.sin(A4_YAW)
        rel = points[:, :2] - A4_CENTER
        local = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], -1)
        on_a4 = table_px & (np.abs(local[:, 0]) <= A4_SIZE[0] / 2) & (np.abs(local[:, 1]) <= A4_SIZE[1] / 2)
        albedo[on_a4] = COLORS["paper"]
        albedo[label == 2] = COLORS["bowl"]
        albedo[label == 3] = COLORS["cube"]
        for k, prim in enumerate(prims):
            albedo[label == 4 + k] = COLORS[prim[-1]]

        # shadows toward the directional light
        lit = np.ones(n)
        shade_idx = np.where(hit)[0]
        origins = points[shade_idx] + 1e-4 * normal[shade_idx]
        light_dirs = np.broadcast_to(LIGHT_DIR, origins.shape)
        blocked = self.occluders(origins, light_dirs, cube_pose, prims)
        lit[shade_idx[blocked]] = 0.0

        ndotl = np.clip(normal @ LIGHT_DIR, 0, 1)
        ndotv = np.abs((normal * self.dirs).sum(-1))
        intensity = 0.42 + 0.55 * ndotl * lit + 0.12 * ndotv
        intensity[~hit] = 1.0
        rgb = np.clip(albedo * intensity[:, None], 0, 1).reshape(HEIGHT, WIDTH, 3)
        cube_mask = (label == 3).reshape(HEIGHT, WIDTH)
        return rgb, cube_mask


# --------------------------------------------------------------- choreography
def ease(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * u)


def keyframe_track(t: float, keys) -> np.ndarray:
    if t <= keys[0][0]:
        return np.asarray(keys[0][1], float)
    for (t0, v0), (t1, v1) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            u = ease((t - t0) / max(t1 - t0, 1e-9))
            return (1 - u) * np.asarray(v0, float) + u * np.asarray(v1, float)
    return np.asarray(keys[-1][1], float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="test_assets/synthetic_pick_place",
                        help="output path stem (writes <stem>.mp4, <stem>_gt.json, <stem>_gt_masks.npz)")
    parser.add_argument("--noise", type=float, default=2.0, help="pixel noise sigma (0-255 scale)")
    parser.add_argument("--max-frames", type=int, default=0, help="debug: render only the first N frames")
    parser.add_argument("--preview-frame", type=int, default=-1, help="debug: write only this frame as <out>_preview.png")
    args = parser.parse_args()

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    caster = RayCaster()

    grip_aperture = CUBE_SIZE + 0.012  # tip keypoints sit ~6 mm outside the contact pads
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if aperture(mid) > grip_aperture else (lo, mid)
    closure_grasp = 0.5 * (lo + hi)

    cube_center0 = np.array([*CUBE_START_XY, CUBE_SIZE / 2])
    bowl_place = np.array([*BOWL_CENTER_XY, BOWL_FLOOR_Z + CUBE_SIZE / 2 + 0.002])
    t_grasp_closed, t_release = 2.5, 5.2
    pinch_keys = [
        (0.0, cube_center0 + [0.02, 0.40, 0.32]),
        (0.6, cube_center0 + [0.02, 0.40, 0.32]),
        (1.4, cube_center0 + [0.0, 0.03, 0.09]),
        (2.0, cube_center0),
        (2.7, cube_center0),
        (3.3, cube_center0 + [0.0, 0.0, 0.12]),
        (4.4, bowl_place + [0.0, 0.0, 0.12]),
        (5.0, bowl_place),
        (5.6, bowl_place),
        (6.4, bowl_place + [-0.04, 0.30, 0.26]),
        (7.0, bowl_place + [-0.04, 0.30, 0.26]),
    ]
    closure_keys = [(0.0, 0.05), (1.4, 0.0), (2.0, 0.0), (t_grasp_closed, closure_grasp),
                    (t_release, closure_grasp), (5.6, 0.0), (7.0, 0.1)]
    yaw_keys = [(0.0, 0.0), (2.7, 0.0), (4.4, np.deg2rad(18)), (7.0, np.deg2rad(18))]

    # cube faces aligned with the thumb-index closing axis at grasp time
    kp_g = hand_keypoints_local(closure_grasp)
    closing = hand_rotation(0.0) @ (kp_g[8] - kp_g[4])
    cube_rot0 = rot_z(float(np.arctan2(closing[1], closing[0])))

    n_frames = int(round(DURATION_S * FPS))
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    writer = cv2.VideoWriter(str(out_stem.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    rng = np.random.default_rng(0)

    gt_frames, masks = [], []
    attach = None
    cube_pos, cube_rot = cube_center0.copy(), cube_rot0.copy()
    for i in range(n_frames):
        t = i / FPS
        closure = float(keyframe_track(t, closure_keys))
        rot = hand_rotation(float(keyframe_track(t, yaw_keys)))
        kp_local = hand_keypoints_local(closure)
        pinch_local = 0.5 * (kp_local[4] + kp_local[8])
        wrist = keyframe_track(t, pinch_keys) - rot @ pinch_local
        kp_world = wrist + kp_local @ rot.T

        if t_grasp_closed <= t < t_release:
            if attach is None:   # rigidly attach the cube to the hand while held
                attach = (rot.T @ (cube_center0 - wrist), rot.T @ cube_rot0)
            cube_pos = wrist + rot @ attach[0]
            cube_rot = rot @ attach[1]
        elif t >= t_release and attach is not None:
            yaw = np.arctan2(cube_rot[1, 0], cube_rot[0, 0])
            cube_pos = np.array([cube_pos[0], cube_pos[1], BOWL_FLOOR_Z + CUBE_SIZE / 2])
            cube_rot = rot_z(yaw)  # settled flat on the bowl floor

        if 0 <= args.preview_frame != i:
            continue
        for attempt in range(30):  # this laptop runs other heavy jobs; ride out transient memory pressure
            try:
                rgb, cube_mask = caster.render((cube_pos, cube_rot), hand_primitives(kp_world, rot, wrist))
                break
            except MemoryError:
                gc.collect()
                time.sleep(5)
        else:
            raise MemoryError("render kept failing for lack of memory")
        if args.preview_frame == i:
            cv2.imwrite(str(out_stem) + "_preview.png", (rgb[..., ::-1] * 255).astype(np.uint8))
            print("wrote preview")
            return
        frame = (rgb[..., ::-1] * 255.0).astype(np.float32)
        frame = cv2.GaussianBlur(frame, (0, 0), 0.6)  # soften ray-cast aliasing a little
        frame += rng.normal(0, args.noise, frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
        masks.append(np.packbits(cube_mask))

        gt_frames.append({
            "t": round(t, 5),
            "hand_keypoints_world": np.round(kp_world, 5).tolist(),
            "pinch_midpoint_world": np.round(0.5 * (kp_world[4] + kp_world[8]), 5).tolist(),
            "thumb_index_aperture_m": round(float(np.linalg.norm(kp_world[4] - kp_world[8])), 5),
            "cube_center_world": np.round(cube_pos, 5).tolist(),
            "cube_yaw_rad": round(float(np.arctan2(cube_rot[1, 0], cube_rot[0, 0])), 5),
            "cube_visible_px": int(cube_mask.sum()),
        })
        if i % 30 == 0:
            print(f"frame {i}/{n_frames}", flush=True)
    writer.release()

    c, s = np.cos(A4_YAW), np.sin(A4_YAW)
    half = np.array(A4_SIZE) / 2
    corners_local = np.array([[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]])
    corners_world = np.array([[A4_CENTER[0] + c * x - s * y, A4_CENTER[1] + s * x + c * y, 0.0] for x, y in corners_local])
    corners_cam = corners_world @ caster.r_cv.T + caster.t_cv
    corners_px = (corners_cam @ caster.K.T)[:, :2] / corners_cam[:, 2:3]

    gt = {
        "description": "Synthetic ray-cast render; world frame: table top z=0, z up, metres.",
        "fps": FPS, "n_frames": n_frames, "width": WIDTH, "height": HEIGHT,
        "camera": {"K": caster.K.tolist(), "R_cv_from_world": caster.r_cv.tolist(), "t_cv": caster.t_cv.tolist(),
                   "fovy_deg": FOVY_DEG, "position_world": CAMERA_POS.tolist()},
        "a4": {"corners_world": corners_world.tolist(), "corners_px": corners_px.tolist(),
               "long_axis_world": [c, s, 0.0], "center_world": [*A4_CENTER, 0.0]},
        "cube_size_m": CUBE_SIZE,
        "bowl_center_world": [*BOWL_CENTER_XY, 0.0],
        "pick_center_world": cube_center0.tolist(),
        "pick_yaw_rad": float(np.arctan2(cube_rot0[1, 0], cube_rot0[0, 0])),
        "place_center_world": gt_frames[-1]["cube_center_world"],
        "grasp_time_s": t_grasp_closed,
        "release_time_s": t_release,
        "closure_at_grasp": closure_grasp,
        "frames": gt_frames,
    }
    Path(str(out_stem) + "_gt.json").write_text(json.dumps(gt))
    np.savez_compressed(str(out_stem) + "_gt_masks.npz", packed=np.stack(masks), shape=np.array([HEIGHT, WIDTH]))
    print(f"wrote {out_stem.with_suffix('.mp4')} ({n_frames} frames), _gt.json, _gt_masks.npz")
    print(f"grasp aperture {aperture(closure_grasp):.4f} m at closure {closure_grasp:.3f}; "
          f"open aperture {aperture(0.0):.4f} m")


if __name__ == "__main__":
    main()
