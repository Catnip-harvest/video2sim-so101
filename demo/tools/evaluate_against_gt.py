"""Score an extractor run on the synthetic clip against its exact ground truth.

    python -I tools/evaluate_against_gt.py --spec outputs/synthetic/task_spec.json \
        --gt test_assets/synthetic_pick_place_gt.json

Everything is compared in the extractor's own human table frame (A4 centre,
+x along the long edge pointing right in the image, +z up), rebuilt from the
ground-truth A4 pose with the same convention. Writes <spec dir>/eval_vs_gt.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    gt = json.loads(Path(args.gt).read_text())
    out_dir = Path(args.spec).parent

    cam = gt["camera"]
    K_gt = np.array(cam["K"])
    R_cw = np.array(cam["R_cv_from_world"])
    t_cw = np.array(cam["t_cv"])
    scale = spec["source_video"]["processing_scale"]

    # ground-truth table frame with the extractor's convention
    a4_c = np.array(gt["a4"]["center_world"])
    x_ax = np.array(gt["a4"]["long_axis_world"])
    z_ax = np.array([0.0, 0, 1])

    def project_world(p):
        c = np.atleast_2d(p) @ R_cw.T + t_cw
        uv = c @ K_gt.T
        return uv[:, :2] / uv[:, 2:3]

    o_px, x_px = project_world(np.array([a4_c, a4_c + 0.1 * x_ax]))
    if x_px[0] < o_px[0]:
        x_ax = -x_ax
    y_ax = np.cross(z_ax, x_ax)
    R_wt = np.column_stack([x_ax, y_ax, z_ax])          # table axes in world

    def to_table(p):
        return (np.atleast_2d(p) - a4_c) @ R_wt

    cal = spec["calibration"]
    res: dict = {}
    # --- calibration ---------------------------------------------------------
    f_est = cal["K_processing_px"][0][0] / scale
    res["focal_px_est_vs_gt"] = [round(f_est, 2), round(float(K_gt[0, 0]), 2)]
    res["focal_error_pct"] = round(100 * (f_est - K_gt[0, 0]) / K_gt[0, 0], 3)
    cam_center_gt = to_table(-R_cw.T @ t_cw)[0]
    cam_center_est = np.array(cal["camera_center_table_m"])
    res["camera_center_error_m"] = round(float(np.linalg.norm(cam_center_est - cam_center_gt)), 4)
    res["camera_height_est_vs_gt_m"] = [cal["camera_height_above_table_m"], round(float(cam_center_gt[2]), 4)]
    n_est = np.array(cal["R_camera_from_table"])[:, 2]
    n_gt = R_cw @ z_ax
    res["table_normal_error_deg"] = round(float(np.rad2deg(np.arccos(np.clip(n_est @ n_gt, -1, 1)))), 3)
    if "a4_corners_px" in cal:
        est = np.array(cal["a4_corners_px"]) / scale
        gtc = np.array(gt["a4"]["corners_px"])
        d = np.linalg.norm(est[:, None] - gtc[None], axis=2)
        res["a4_corner_error_px_mean_max"] = [round(float(d.min(1).mean()), 3), round(float(d.min(1).max()), 3)]

    # --- object ----------------------------------------------------------------
    human = spec["human_table_frame"]
    pick_gt = to_table(gt["pick_center_world"])[0]
    place_gt = to_table(gt["place_center_world"])[0]
    pick_est = np.array(human["object_start_pose_xyz_yaw"][:3])
    place_est = np.array(human["place_target_xyz"])
    res["pick_error_m"] = round(float(np.linalg.norm(pick_est - pick_gt)), 4)
    res["pick_error_xyz_m"] = np.round(pick_est - pick_gt, 4).tolist()
    res["place_error_m"] = round(float(np.linalg.norm(place_est - place_gt)), 4)
    res["place_error_xyz_m"] = np.round(place_est - place_gt, 4).tolist()
    res["object_size_est_vs_gt_m"] = [spec["object"]["size_m"], gt["cube_size_m"]]

    times = np.array([f["t"] for f in gt["frames"]])
    cube = to_table(np.array([f["cube_center_world"] for f in gt["frames"]]))
    ow = np.array(human["object_waypoints"])
    gt_at = np.column_stack([np.interp(ow[:, 3], times, cube[:, j]) for j in range(3)])
    err = np.linalg.norm(ow[:, :3] - gt_at, axis=1)
    moving = (ow[:, 3] > gt["grasp_time_s"]) & (ow[:, 3] < gt["release_time_s"])
    res["object_path_error_m_mean_max_all"] = [round(float(err.mean()), 4), round(float(err.max()), 4)]
    if moving.any():
        res["object_path_error_m_mean_max_carried"] = [round(float(err[moving].mean()), 4),
                                                       round(float(err[moving].max()), 4)]
    moved = np.where(np.linalg.norm(cube - cube[0], axis=1) > 0.003)[0]
    lift_gt = float(times[moved[0] - 1]) if len(moved) else float("nan")   # last frame at rest
    # set-down: from here on the cube stays within 3 mm of its final rest pose
    dist_final = np.linalg.norm(cube - cube[-1], axis=1)
    far = np.where(dist_final > 0.003)[0]
    set_gt = float(times[far[-1] + 1]) if len(far) else float(times[0])
    res["timing_s_est_vs_gt"] = {
        "grasp": [spec["grasp_time"], gt["grasp_time_s"]],
        "lift_off": [spec["lift_off_time"], lift_gt],
        "set_down": [spec["set_down_time"], round(set_gt, 3)],
        "release": [spec["release_time"], gt["release_time_s"]],
    }
    yaw_gt = gt["pick_yaw_rad"] - np.arctan2(x_ax[1], x_ax[0])
    yaw_est = human["object_start_pose_xyz_yaw"][3]
    dyaw = (yaw_est - yaw_gt + np.pi / 2) % np.pi - np.pi / 2       # a box yaw is defined mod 180 deg
    res["grasp_yaw_error_deg_mod180"] = round(float(np.rad2deg(dyaw)), 2)

    # --- masks -------------------------------------------------------------------
    gt_masks_path = Path(args.gt.replace("_gt.json", "_gt_masks.npz"))
    est_masks_path = out_dir / "object_masks.npz"
    if gt_masks_path.exists() and est_masks_path.exists():
        g = np.load(gt_masks_path)
        e = np.load(est_masks_path)
        gh, gw = g["shape"]
        eh, ew = e["shape"]
        src_idx = np.round(np.array(spec_times(spec, out_dir)) * gt["fps"]).astype(int)
        ious = []
        for k, si in enumerate(src_idx):
            if si >= len(g["packed"]):
                continue
            gm = np.unpackbits(g["packed"][si])[: gh * gw].reshape(gh, gw).astype(bool)
            em = np.unpackbits(e["packed"][k])[: eh * ew].reshape(eh, ew).astype(bool)
            if (eh, ew) != (gh, gw):
                import cv2
                em = cv2.resize(em.astype(np.uint8), (int(gw), int(gh)), interpolation=cv2.INTER_NEAREST).astype(bool)
            union = (gm | em).sum()
            if union:
                ious.append((gm & em).sum() / union)
        res["object_mask_iou_mean_min"] = [round(float(np.mean(ious)), 4), round(float(np.min(ious)), 4)]

    (out_dir / "eval_vs_gt.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def spec_times(spec, out_dir: Path):
    dbg = json.loads((out_dir / spec["files"]["trajectory_debug"]).read_text())
    return [f["t"] for f in dbg["frames"]]


if __name__ == "__main__":
    main()
