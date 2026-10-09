"""Convert the extractor's task_spec.json into the SO-101 mjlab task's own format
(`so101_pickplace_task_spec/v2`, parsed by manip/task/so101_pickplace/task_spec.py).

    python -I demo_extractor/so101_export.py outputs/<run>/task_spec.json   (extract_demo.py also writes it)

Both files use the same robot base frame (origin at the base on the table, +x
forward, +y left, +z up, metres, demo-clock seconds), so this is a re-labelling plus
four adaptations the task needs:
  * object size is clamped into the 20-50 mm band the gripper was tested for (a
    warning is recorded; the real object should then be swapped for one in band);
  * start / place become small ranges around the measured poses (+-1 cm, yaw +-0.3);
  * the object trajectory is padded with rest samples so that
    t0 <= grasp_time <= release_time <= t_end, as the task's parser requires;
  * heights are re-referenced to the clamped object (a resting object's centre sits
    at half its height).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FORMAT = "so101_pickplace_task_spec/v2"
SIZE_BAND = (0.020, 0.050)


def convert(spec: dict, spec_path: str | None = None, pos_tol: float = 0.01, yaw_tol: float = 0.3) -> tuple[dict, list[str]]:
    notes: list[str] = []
    obj = spec["object"]
    size_m = np.asarray(obj["size_m"], float)
    measured = float(max(size_m[0], size_m[2]))
    size = float(np.clip(measured, *SIZE_BAND))
    if abs(size - measured) > 1e-6:
        notes.append(f"object size {measured * 1000:.1f} mm clamped to {size * 1000:.0f} mm "
                     f"(task band {SIZE_BAND[0] * 1000:.0f}-{SIZE_BAND[1] * 1000:.0f} mm)")
    sx, sy, sz, syaw = obj["start_pose_xyz_yaw"]
    px, py, pz = spec["place_target_xyz"]
    sep = float(np.hypot(px - sx, py - sy))
    if sep < 0.11:
        notes.append(f"pick-place separation {sep:.3f} m is below the task's 0.11 m minimum")

    wp = np.asarray(spec["object_waypoints"], float)          # x, y, z, t (robot frame)
    dz = size / 2 - float(sz)                                  # re-reference heights to the clamped size
    wp[:, 2] = np.maximum(wp[:, 2] + dz, 0.0)
    grasp, release = float(spec["grasp_time"]), float(spec["release_time"])
    if grasp < wp[0, 3]:
        wp = np.vstack([[wp[0, 0], wp[0, 1], wp[0, 2], grasp - 0.2], wp])
    if release > wp[-1, 3]:
        wp = np.vstack([wp, [wp[-1, 0], wp[-1, 1], wp[-1, 2], release + 0.2]])
    release = max(release, grasp)

    out = {
        "format": FORMAT,
        "frame": "robot_base",
        "units": "metres, seconds, radians",
        "source": {
            "kind": "human_video",
            "video": spec["source_video"]["path"],
            "extractor_spec": spec_path,
            "calibration_method": spec["calibration"]["method"],
            "notes": "Exported from the watch-me-once extractor (object-centric). Waypoints are the OBJECT "
                     "centre on the demo clock; grasp/release times and approach side come from the hand. "
                     + " ".join(notes),
            "extractor_warnings": spec["quality"].get("warnings", []),
        },
        "mount_height": 0.0,
        "object": {
            "kind": "box",
            "size": round(size, 4),
            "mass": 0.025,
            "start_pose_range": {"x": [round(sx - pos_tol, 4), round(sx + pos_tol, 4)],
                                 "y": [round(sy - pos_tol, 4), round(sy + pos_tol, 4)],
                                 "yaw": [round(syaw - yaw_tol, 4), round(syaw + yaw_tol, 4)]},
        },
        "place_target": {
            "kind": "bin",
            "pos_range": {"x": [round(px - pos_tol, 4), round(px + pos_tol, 4)],
                          "y": [round(py - pos_tol, 4), round(py + pos_tol, 4)]},
            "min_separation": 0.11,
            "success_xy_margin": 0.005,
        },
        "approach": {"direction": [round(float(v), 4) for v in spec["approach_dir"]], "weight": 0.25,
                     "gate_std": 0.06},
        "object_trajectory": {
            "waypoints": np.round(wp, 4).tolist(),
            "grasp_time": round(grasp, 4),
            "release_time": round(release, 4),
            "robot_time_scale": 2.0,
            "smoothing_window_s": 0.25,
            "resample_dt": 0.1,
            "weight": 4.0,
            "std": 0.05,
            "time_slack_s": 1.0,
        },
    }
    return out, notes


def main() -> None:
    src = Path(sys.argv[1])
    spec = json.loads(src.read_text(encoding="utf-8"))
    out, notes = convert(spec, str(src.resolve()))
    dst = src.with_name("so101_task_spec.json")
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dst}")
    for n in notes:
        print(f"note: {n}")


if __name__ == "__main__":
    main()
