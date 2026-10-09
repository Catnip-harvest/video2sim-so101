"""Human table frame -> robot table frame.

A person's tabletop workspace is bigger than a small SO-101's (~0.3 m reach),
so positions go through a planar similarity transform (official term: a 2D
similarity, i.e. rotation + uniform scale + offset in the table plane):

    p_robot.xy = s * R(yaw) * p_human.xy + offset
    p_robot.z  = z_scale * p_human.z          (default z_scale = 1: heights kept)

Object size and gripper width are physical and are never scaled.

Robot table frame: origin at the robot base on the table surface, +x forward,
+y left, +z up. Default fit: pick on the robot's right, place on its left, both
at the same radius inside a 0.25 m half-disc in front of the base; shrink until
every path point is reachable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WorkspaceMap:
    scale: float
    yaw: float
    offset_xy: np.ndarray
    z_scale: float = 1.0

    def point(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p).astype(float)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        xy = self.scale * p[:, :2] @ np.array([[c, s], [-s, c]]) + self.offset_xy
        return np.column_stack([xy, self.z_scale * p[:, 2]])

    def direction(self, v: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        out = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])
        return out / (np.linalg.norm(out) + 1e-12)

    def yaw_angle(self, a: float) -> float:
        return float(np.arctan2(np.sin(a + self.yaw), np.cos(a + self.yaw)))

    def as_dict(self) -> dict:
        return {"scale_xy": round(self.scale, 5), "yaw_rad": round(self.yaw, 5),
                "offset_xy_m": np.round(self.offset_xy, 5).tolist(), "z_scale": self.z_scale,
                "formula": "xy_robot = scale_xy * Rz(yaw) @ xy_human + offset_xy; z_robot = z_scale * z_human"}


def fit_workspace(pick_xy: np.ndarray, place_xy: np.ndarray, path_xy: np.ndarray, reach: float = 0.25,
                  min_radius: float = 0.10, scale=None, yaw=None, offset=None, z_scale: float = 1.0,
                  min_span: float = 0.12) -> tuple[WorkspaceMap, dict]:
    """Default similarity transform, overridable piece by piece from the CLI."""
    span = place_xy - pick_xy
    d = float(np.linalg.norm(span))
    if yaw is None:   # pick -> place becomes the robot's right -> left (+y)
        yaw = float(np.pi / 2 - np.arctan2(span[1], span[0])) if d > 1e-6 else 0.0
    r_mid = 0.5 * (reach + min_radius) + 0.02          # 0.195 m for the defaults
    chord_max = 2 * r_mid * np.sin(np.deg2rad(55))
    # min_span: the SO-101 task needs pick and place >= 0.11 m apart (its bin would
    # overlap the object otherwise), so short human moves are scaled UP to min_span.
    s_min = min_span / d if d > 1e-6 else 1.0
    s = float(scale) if scale is not None else (max(s_min, min(1.0, chord_max / d)) if d > 1e-6 else 1.0)
    info = {"robot_reach_m": reach, "robot_min_radius_m": min_radius, "human_pick_place_distance_m": round(d, 4),
            "min_pick_place_span_m": min_span}

    def build(s_val: float) -> WorkspaceMap:
        if offset is not None:
            off = np.asarray(offset, float)
        else:
            mid = 0.5 * (pick_xy + place_xy)
            half = 0.5 * s_val * d
            x_c = np.sqrt(max(r_mid ** 2 - half ** 2, (min_radius + 0.02) ** 2))
            c, sn = np.cos(yaw), np.sin(yaw)
            off = np.array([x_c, 0.0]) - s_val * np.array([c * mid[0] - sn * mid[1], sn * mid[0] + c * mid[1]])
        return WorkspaceMap(scale=s_val, yaw=yaw, offset_xy=off, z_scale=z_scale)

    wmap = build(s)
    for _ in range(25):
        pts = wmap.point(np.column_stack([path_xy, np.zeros(len(path_xy))]))[:, :2]
        r = np.linalg.norm(pts, axis=1)
        ok = (r <= reach) & (r >= min_radius) & (pts[:, 0] >= 0.02)
        if ok.all() or scale is not None or s * 0.92 < s_min:
            break
        s *= 0.92
        wmap = build(s)
    pts = wmap.point(np.column_stack([path_xy, np.zeros(len(path_xy))]))[:, :2]
    r = np.linalg.norm(pts, axis=1)
    info.update({"fit_ok": bool(((r <= reach + 1e-6) & (r >= min_radius - 1e-6) & (pts[:, 0] >= 0.02 - 1e-6)).all()),
                 "path_radius_min_max_m": [round(float(r.min()), 4), round(float(r.max()), 4)]})
    return wmap, info
