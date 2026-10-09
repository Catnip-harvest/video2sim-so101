"""Debug overlay video and 3D trajectory plot."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .hands import HAND_CONNECTIONS
from .video_io import VideoWriter

PHASE_COLORS = {"approach": (200, 200, 200), "grasp": (0, 200, 255), "transport": (0, 220, 0),
                "release": (255, 120, 0), "retreat": (180, 120, 255)}


def phase_at(phases: list[dict], t: float) -> str:
    for ph in phases:
        if ph["t_start"] <= t < ph["t_end"] or (ph is phases[-1] and t >= ph["t_start"]):
            return ph["name"]
    return phases[0]["name"]


def write_overlay(path: Path, frames, times, calib, hand_px, track, geom, a4_corners=None) -> None:
    h, w = frames[0].shape[:2]
    fps = 1.0 / float(np.median(np.diff(times))) if len(times) > 1 else 15.0
    writer = VideoWriter(path, fps, (w, h))
    axes_pts = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    axes_px = calib.project_table(axes_pts)
    path_px = calib.project_table(geom.dense_xyz)
    rejected_frames = {k for frames_ in geom.rejected.values() for k in frames_}
    for k, frame in enumerate(frames):
        img = frame.copy()
        mask = track.masks[k]
        if mask is not None and mask.any():
            tint = img.copy()
            tint[mask] = (0.45 * tint[mask] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
            img = tint
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255, 0, 255), 1)
        if a4_corners is not None:
            cv2.polylines(img, [np.round(a4_corners).astype(np.int32)], True, (0, 255, 255), 1)
            for c in a4_corners:
                cv2.circle(img, tuple(np.round(c).astype(int)), 4, (0, 255, 255), -1)
        o = tuple(np.round(axes_px[0]).astype(int))
        for end, color in zip(axes_px[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            cv2.arrowedLine(img, o, tuple(np.round(end).astype(int)), color, 2, tipLength=0.15)
        cv2.polylines(img, [np.round(path_px).astype(np.int32)], False, (0, 165, 255), 1)
        if np.isfinite(hand_px[k, 0, 0]):
            pts = np.round(hand_px[k]).astype(int)
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 1)
            for p in pts:
                cv2.circle(img, tuple(p), 2, (0, 0, 255), -1)
            mid = tuple(np.round(0.5 * (hand_px[k, 4] + hand_px[k, 8])).astype(int))
            cv2.circle(img, mid, 4, (0, 255, 0), 1)
        if np.isfinite(geom.object_xyz_raw[k, 0]):
            c = calib.project_table(geom.object_xyz_raw[k][None])[0]
            cv2.drawMarker(img, tuple(np.round(c).astype(int)), (0, 165, 255), cv2.MARKER_CROSS, 12, 2)
        phase = phase_at(geom.phases, times[k])
        cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
        label = f"t={times[k]:5.2f}s  phase: {phase}"
        if k in rejected_frames:
            label += "  [object sample rejected]"
        ap = geom.aperture_m[k]
        if np.isfinite(ap):
            label += f"  pinch {100 * ap:4.1f} cm"
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, PHASE_COLORS.get(phase, (255, 255, 255)), 1,
                    cv2.LINE_AA)
        writer.write(img)
    writer.close()


def plot_trajectories(path: Path, geom, spec_robot: dict, wmap, a4_size=(0.297, 0.210)) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_title("Human table frame (A4-centred)")
    hw, hh = a4_size[0] / 2, a4_size[1] / 2
    ax.plot([-hw, hw, hw, -hw, -hw], [-hh, -hh, hh, hh, -hh], [0, 0, 0, 0, 0], color="0.6", lw=1, label="A4 sheet")
    ok = np.isfinite(geom.pinch_table[:, 0])
    ax.plot(*geom.pinch_table[ok].T, color="0.75", lw=1, label="hand pinch (debug, not a target)")
    raw = geom.object_xyz_raw
    okr = np.isfinite(raw[:, 0])
    ax.scatter(*raw[okr].T, s=8, color="tab:orange", label="object samples kept")
    ax.plot(*geom.dense_xyz.T, color="tab:red", lw=2, label="object path (smoothed, dense)")
    ax.scatter(*geom.object_est.start_xyz, color="green", s=60, marker="o", label="pick (object start)")
    ax.scatter(*geom.place_xyz, color="blue", s=60, marker="^", label="place")
    _equal_axes(ax, np.vstack([geom.dense_xyz, [[-hw, -hh, 0], [hw, hh, 0]]]))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend(fontsize=7, loc="upper left")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.set_title(f"Robot table frame (scale {wmap.scale:.2f})")
    reach = spec_robot["workspace_mapping"]["robot_reach_m"]
    ang = np.linspace(-np.pi / 2, np.pi / 2, 60)
    ax2.plot(reach * np.cos(ang), reach * np.sin(ang), 0 * ang, color="0.6", lw=1, label=f"{reach} m reach")
    ax2.scatter([0], [0], [0], color="k", s=40, marker="s", label="robot base")
    ow = np.array(spec_robot["object_waypoints"])
    ax2.plot(ow[:, 0], ow[:, 1], ow[:, 2], color="tab:red", lw=2, label="object_waypoints")
    wp = np.array(spec_robot["waypoints"])
    ax2.scatter(wp[:, 0], wp[:, 1], wp[:, 2], color="k", s=14, label="suggested gripper waypoints")
    start = spec_robot["object"]["start_pose_xyz_yaw"]
    ax2.scatter(*start[:3], color="green", s=60, marker="o", label="pick")
    ax2.scatter(*spec_robot["place_target_xyz"], color="blue", s=60, marker="^", label="place")
    ad = np.array(spec_robot["approach_dir"])
    ax2.quiver(*(np.array(start[:3]) - 0.08 * ad), *(0.08 * ad), color="purple", label="approach_dir")
    _equal_axes(ax2, np.vstack([ow[:, :3], [[0, -reach, 0], [reach, reach, 0]]]))
    ax2.set_xlabel("x fwd [m]"); ax2.set_ylabel("y left [m]"); ax2.set_zlabel("z [m]")
    ax2.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _equal_axes(ax, pts: np.ndarray) -> None:
    lo, hi = pts.min(0), pts.max(0)
    center = 0.5 * (lo + hi)
    r = 0.5 * float(np.max(hi - lo)) + 0.02
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(max(0.0, center[2] - r), center[2] + r)
