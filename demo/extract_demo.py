"""Watch me once: one third-person phone video of a person picking up a small
object and placing it somewhere -> an object-centric task spec for a two-finger
parallel-gripper arm (LeRobot SO-101).

    .venv/Scripts/python.exe -I extract_demo.py VIDEO.mp4 --out outputs/my_demo

Writes into --out:
    task_spec.json         the contract consumed by the mjlab RL task (see README)
    trajectory_debug.json  per-frame measurements (object samples, hand pinch, rejections)
    debug_overlay.mp4      hand skeleton, object mask, A4 corners, table axes, phase labels
    trajectories_3d.png    object path in the human table frame and the robot frame
    so101_task_spec.json   the same demo in the mjlab task's format (so101_pickplace_task_spec/v2)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # run with -I: the script dir is not on sys.path by default

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from demo_extractor import calibration as cal  # noqa: E402
from demo_extractor.geometry import analyse_demo  # noqa: E402
from demo_extractor.hands import (choose_world_z_sign, detect_hands, fill_and_smooth,  # noqa: E402
                                  lift_to_camera)
from demo_extractor.video_io import load_video  # noqa: E402
from demo_extractor.workspace import fit_workspace  # noqa: E402

THIRD_PARTY = HERE / "third_party"
SCHEMA_VERSION = "watch_me_once.task_spec/2 (object-centric)"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def floats(text: str | None, n: int | None = None) -> list[float] | None:
    if text is None:
        return None
    vals = [float(v) for v in text.replace(" ", "").split(",") if v]
    if n is not None and len(vals) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated numbers, got {text!r}")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--out", default=None, help="output directory (default outputs/<video name>)")
    g = p.add_argument_group("object prompt (frame 0, source-video pixels); default: auto = object nearest the hand at grasp")
    g.add_argument("--object-point", help="x,y click on the object in frame 0")
    g.add_argument("--object-box", help="x0,y0,x1,y1 box around the object in frame 0")
    g.add_argument("--object-size-m", help="known object size: w,d,h in metres (else estimated from the mask)")
    g = p.add_argument_group("calibration")
    g.add_argument("--a4-corners", help="x1,y1,...,x4,y4 A4 corners in source pixels (else auto-detected)")
    g.add_argument("--no-a4", action="store_true", help="no A4 sheet in view: use the hand-size + depth-model fallback")
    g.add_argument("--hfov-deg", type=float, default=65.0,
                   help="horizontal field of view used when focal length cannot be measured (phone 1x ~ 65)")
    g.add_argument("--focal-px", type=float, help="known focal length in source pixels (overrides estimation)")
    g.add_argument("--camera-pitch-deg", type=float,
                   help="no-A4 fallback only: camera looks down this many degrees, horizon level (e.g. from the "
                        "axis ratio of a round rim or liquid surface: ray elevation = asin(minor/major))")
    g = p.add_argument_group("robot")
    g.add_argument("--gripper-max-open", type=float, default=0.06, help="max gripper opening, m (placeholder)")
    g.add_argument("--robot-reach", type=float, default=0.25, help="radius of the reachable half-disc, m")
    g.add_argument("--robot-min-radius", type=float, default=0.10, help="keep-out radius around the base, m")
    g.add_argument("--robot-scale", type=float, help="override the xy scale of the human->robot map")
    g.add_argument("--robot-yaw-deg", type=float, help="override the yaw of the human->robot map")
    g.add_argument("--robot-offset", help="override the xy offset of the human->robot map: x,y in metres")
    g.add_argument("--robot-z-scale", type=float, default=1.0, help="height scale (default 1: heights kept)")
    g = p.add_argument_group("processing")
    g.add_argument("--max-fps", type=float, default=15.0, help="process at most this many frames per second")
    g.add_argument("--max-side", type=int, default=960, help="downscale so the longer image side is at most this")
    g.add_argument("--dense-hz", type=float, default=30.0, help="sample rate of object_waypoints")
    g.add_argument("--pinch-mode", choices=["thumb-index", "thumb-index-middle"], default="thumb-index")
    g.add_argument("--hand-backend", choices=["auto", "mediapipe", "onnx"], default="auto",
                   help="MediaPipe Tasks, or the same MediaPipe models as ONNX via OpenCV DNN")
    g.add_argument("--skip-depth", action="store_true", help="skip the Depth Anything V2 cross-check")
    g.add_argument("--reuse-object-masks", action="store_true",
                   help="reuse <out>/object_masks.npz from a previous run instead of re-running SAM 2")
    g.add_argument("--device", default="cpu", help="torch device for SAM 2 / depth (cpu or cuda)")
    return p.parse_args()


def a4_rejection_reason(calib, a4_info: dict) -> str | None:
    """A bright quadrilateral is not necessarily the sheet: check it behaves like one."""
    info = calib.info
    if info["a4_reprojection_rms_px"] > 2.0:
        return f"corner reprojection {info['a4_reprojection_rms_px']:.1f} px > 2 px"
    if calib.focal_source == "hfov_prior" and abs(info["a4_aspect_ratio_implied"] - 297 / 210) > 0.12:
        return f"implied aspect {info['a4_aspect_ratio_implied']:.2f}, A4 is 1.41"
    if info["camera_height_above_table_m"] < 0.05:
        return "camera would be below 5 cm above the table"
    if a4_info.get("a4_frames_used", 2) < 2:
        return "sheet found in fewer than 2 frames"
    return None


def grasp_candidate_from_hand(aperture: np.ndarray, hand_ok: np.ndarray) -> int | None:
    """First moment the thumb-index gap closes after having been open."""
    idx = np.where(hand_ok)[0]
    if len(idx) < 5:
        return None
    a = aperture[idx]
    lo, hi = np.percentile(a, 10), np.percentile(a, 90)
    if hi - lo < 1e-4:
        return None
    closed = a < lo + 0.3 * (hi - lo)
    opened = a > lo + 0.6 * (hi - lo)
    seen_open = False
    for j in range(len(idx) - 2):
        seen_open |= bool(opened[j])
        if seen_open and closed[j] and closed[j + 1] and closed[j + 2]:
            return int(idx[min(j + 1, len(idx) - 1)])
    return None


def wait_for_memory(min_free_gb: float = 2.5, timeout_s: float = 600) -> None:
    """This laptop hit 'paging file is too small' with several torch jobs at once."""
    try:
        import psutil
    except ImportError:
        return
    start = time.time()
    while psutil.virtual_memory().available / 1e9 < min_free_gb and time.time() - start < timeout_s:
        log(f"waiting for memory: {psutil.virtual_memory().available / 1e9:.1f} GB free")
        time.sleep(15)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out) if args.out else HERE / "outputs" / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    clip = load_video(args.video, max_side=args.max_side, max_fps=args.max_fps)
    sc = clip.scale
    w, h = clip.size
    log(f"video {clip.path}: {len(clip.frames)} frames at {clip.processed_fps:.1f} fps, {w}x{h} "
        f"(source {clip.source_size[0]}x{clip.source_size[1]} @ {clip.source_fps:.2f} fps)")

    # 1. hand keypoints ------------------------------------------------------
    track_h, hand_backend = detect_hands(clip.frames, clip.times, THIRD_PARTY, backend=args.hand_backend)
    log(f"hand detected in {track_h.detected.mean():.0%} of frames ({hand_backend})")

    # camera must stay still: compare hand-free first and last frames
    from demo_extractor.object_track import camera_shift_px
    free = [k for k in range(len(clip.frames)) if not track_h.detected[k]]
    k_a = free[0] if free else 0
    k_b = free[-1] if free else len(clip.frames) - 1
    cam_shift = camera_shift_px(clip.frames[k_a], clip.frames[k_b])
    camera_motion_px = float(np.hypot(*cam_shift))
    if camera_motion_px > 3.0:
        warnings.append(f"camera moved ~{camera_motion_px:.0f} px between hand-free frames {k_a} and {k_b}: "
                        "calibration and the object path assume a still camera")

    # 2. calibration ------------------------------------------------------------
    focal_px = args.focal_px * sc if args.focal_px else None
    calib = None
    a4_info: dict = {}
    corners = None
    if not args.no_a4:
        if args.a4_corners:
            corners = np.array(floats(args.a4_corners, 8)).reshape(4, 2) * sc
            corners = cal._order_quad(corners)
            a4_info["a4_source"] = "cli"
        else:
            corners, a4_info = cal.detect_a4_over_frames(clip.frames)
            a4_info["a4_source"] = "auto_detect"
        if corners is not None:
            calib = cal.calibrate_from_a4(corners, (w, h), args.hfov_deg, focal_px)
            calib.info.update(a4_info)
            log(f"A4 calibration: f={calib.K[0, 0]:.1f}px ({calib.focal_source}), camera height "
                f"{calib.info['camera_height_above_table_m']:.3f} m, reproj {calib.info['a4_reprojection_rms_px']:.3f}px")
            reason = a4_rejection_reason(calib, a4_info)
            if reason and a4_info.get("a4_source") != "cli":
                warnings.append(f"A4 candidate rejected ({reason}); using the no-A4 fallback")
                log(f"A4 candidate rejected: {reason}")
                calib, corners = None, None
        if calib is None and corners is None and not any("A4 candidate rejected" in m for m in warnings):
            warnings.append("A4 sheet not found: falling back to hand-size + depth-model calibration")
            log("A4 sheet not found -> fallback calibration")

    f_guess = focal_px or (w / 2) / np.tan(np.deg2rad(args.hfov_deg) / 2)
    K_guess = np.array([[f_guess, 0, w / 2], [0, f_guess, h / 2], [0, 0, 1.0]]) if calib is None else calib.K
    z_sign = choose_world_z_sign(track_h, K_guess)
    hand_cam_raw, hand_rms = lift_to_camera(track_h, K_guess, z_sign)
    hand_cam_raw = fill_and_smooth(hand_cam_raw, max_gap=int(round(0.3 * clip.processed_fps)))
    hand_px = fill_and_smooth(np.concatenate([track_h.image_px, np.zeros((len(clip.frames), 21, 1))], -1),
                              max_gap=int(round(0.3 * clip.processed_fps)))[..., :2]
    hand_ok = np.isfinite(hand_cam_raw[:, 0, 0])
    aperture_raw = np.linalg.norm(hand_cam_raw[:, 4] - hand_cam_raw[:, 8], axis=1)
    k_cand = grasp_candidate_from_hand(aperture_raw, hand_ok)

    # 3. object track (SAM 2) ----------------------------------------------------
    from demo_extractor.object_track import ObjectTrack, Sam2Tracker, build_prompt, change_blob_prompt
    mask_cache = out_dir / "object_masks.npz"
    if args.reuse_object_masks and mask_cache.exists():
        cached = np.load(mask_cache, allow_pickle=False)
        mh, mw = cached["shape"]
        masks = [np.unpackbits(m)[: mh * mw].reshape(mh, mw).astype(bool) for m in cached["packed"]]
        track_o = ObjectTrack(masks=masks, scores=cached["logits"], prompt=json.loads(str(cached["prompt"])))
        log(f"reusing object masks from {mask_cache}")
    else:
        wait_for_memory()
        log("loading SAM 2.1 hiera-tiny")
        tracker = Sam2Tracker(THIRD_PARTY / "sam2.1_hiera_tiny", device=args.device)
        if args.object_point:
            prompt = build_prompt("point", clip.frames, tracker, point=[v * sc for v in floats(args.object_point, 2)])
        elif args.object_box:
            prompt = build_prompt("box", clip.frames, tracker, box=[v * sc for v in floats(args.object_box, 4)])
        else:
            pinch = None if k_cand is None else 0.5 * (hand_px[k_cand, 4] + hand_px[k_cand, 8])
            prompt = None if k_cand is None else change_blob_prompt(clip.frames, hand_px, k_cand, pinch)
            if prompt is None:   # moving camera or no clean change: ask SAM 2 at the grasp, track back
                prompt = build_prompt("auto", clip.frames, tracker, grasp_frame=k_cand, grasp_pinch_px=pinch,
                                      hand_px_at_grasp=None if k_cand is None else hand_px[k_cand])
        log(f"object prompt: { {k: v for k, v in prompt.items() if k != 'mask'} }")
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip.frames]
        masks, logits = tracker.track(frames_rgb, points=prompt["points"], labels=prompt["labels"],
                                      box=prompt["box"], mask=prompt.get("mask"), progress=log)
        prompt = {k: v for k, v in prompt.items() if k != "mask"}
        tracker.close()
        del tracker, frames_rgb
        gc.collect()
        track_o = ObjectTrack(masks=masks, scores=logits, prompt=json.loads(json.dumps(prompt, default=float)))
        np.savez_compressed(mask_cache, shape=np.array([h, w]), logits=logits,
                            packed=np.stack([np.packbits(m) for m in masks]),
                            prompt=np.array(json.dumps(track_o.prompt)))

    # 4. no-A4 fallback: table plane from the depth model + the hand's metric size
    depth_model = None
    if calib is None:
        from demo_extractor.depth import DepthModel
        wait_for_memory()
        depth_model = DepthModel(THIRD_PARTY / "depth_anything_v2_small", device=args.device)
        calib = fallback_calibration(clip, depth_model, K_guess, hand_cam_raw, track_h, track_o, k_cand,
                                     "cli" if focal_px else "hfov_prior",
                                     floats(args.object_size_m, 3) if args.object_size_m else None,
                                     args.camera_pitch_deg)
        scale_src = "the known object size" if calib.info.get("fallback_metric_scale_source") ==             "known_object_size" else "an average hand"
        warnings.append(f"calibration is the no-A4 fallback: metric scale from {scale_src}, focal length from "
                        f"--hfov-deg, table tilt from {calib.info.get('fallback_tilt_source')}; "
                        "expect several-cm errors")

    # 5. optional depth cross-check during transport -----------------------------
    depth_check = None
    depth_quality: dict = {}
    if not args.skip_depth:   # aligned to whichever table plane the calibration found
        depth_check, depth_quality = depth_cross_check(clip, calib, track_o, hand_px, args.device, depth_model)
    if depth_model is not None:
        depth_model.close()
    gc.collect()

    # 6. geometry --------------------------------------------------------------
    geom = analyse_demo(clip.times, calib, hand_px, hand_cam_raw, track_o, (w, h),
                        object_size=floats(args.object_size_m, 3) if args.object_size_m else None,
                        pinch_mode=args.pinch_mode, gripper_max_open=args.gripper_max_open,
                        dense_hz=args.dense_hz, depth_cross_check=depth_check)
    if depth_check:
        hand_dist = {}
        for k, d in depth_check.items():
            if hand_ok[k] and np.isfinite(geom.object_xyz_raw[k, 0]):
                cam = calib.table_to_cam(geom.object_xyz_raw[k][None])[0]
                hand_dist[k] = abs(np.linalg.norm(cam) - d) / np.linalg.norm(cam)
        if hand_dist:
            depth_quality["depth_model_vs_hand_object_distance_median_rel_err"] = round(float(np.median(list(hand_dist.values()))), 4)
    log(f"grasp {geom.events['grasp_time']:.2f}s, lift {geom.events['lift_off_time']:.2f}s, set-down "
        f"{geom.events['set_down_time']:.2f}s, release {geom.events['release_time']:.2f}s, hand scale {geom.hand_scale:.3f}")

    # 7. workspace mapping + spec -----------------------------------------------
    wmap, ws_info = fit_workspace(
        geom.object_est.start_xyz[:2], geom.place_xyz[:2], geom.dense_xyz[:, :2], reach=args.robot_reach,
        min_radius=args.robot_min_radius, scale=args.robot_scale,
        yaw=None if args.robot_yaw_deg is None else np.deg2rad(args.robot_yaw_deg),
        offset=floats(args.robot_offset, 2) if args.robot_offset else None, z_scale=args.robot_z_scale)
    spec = build_spec(args, clip, calib, geom, wmap, ws_info, track_o, hand_rms, z_sign, warnings + geom.warnings,
                      depth_quality, hand_backend, camera_motion_px)
    (out_dir / "task_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    from demo_extractor.so101_export import convert
    so101_spec, so101_notes = convert(spec, str((out_dir / "task_spec.json").resolve()))
    (out_dir / "so101_task_spec.json").write_text(json.dumps(so101_spec, indent=2), encoding="utf-8")
    for note_text in so101_notes:
        log(f"so101 export: {note_text}")
    write_debug(out_dir / "trajectory_debug.json", clip, geom, track_o, hand_ok)
    from demo_extractor.viz import plot_trajectories, write_overlay
    write_overlay(out_dir / "debug_overlay.mp4", clip.frames, clip.times, calib, hand_px, track_o, geom,
                  a4_corners=corners)
    plot_trajectories(out_dir / "trajectories_3d.png", geom, spec, wmap)
    log(f"wrote {out_dir / 'task_spec.json'}")
    for wmsg in spec["quality"]["warnings"]:
        log(f"WARNING: {wmsg}")


# ------------------------------------------------------------------ helpers
def fallback_calibration(clip, depth_model, K, hand_cam_raw, track_h, track_o, k_cand, focal_source,
                         object_size=None, camera_pitch_deg=None):
    """No A4: table normal from a plane fit in the depth model's disparity
    (assumes zero disparity shift), height from the hand at the grasp (pinch
    ~2 cm above the table, average-hand metric scale)."""
    k0 = 0
    disp = depth_model.disparity(clip.frames[k0])
    h, w = disp.shape
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    xn, yn = (uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1]
    sel = np.zeros_like(disp, bool)
    sel[int(0.55 * h):, :] = True
    if track_o.masks[k0] is not None:
        sel &= ~cv2.dilate(track_o.masks[k0].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    A = np.column_stack([xn[sel], yn[sel], np.ones(sel.sum())])
    d = disp[sel]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(300):
        i = rng.choice(len(d), 3, replace=False)
        try:
            coef = np.linalg.solve(A[i], d[i])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(A @ coef - d) < 0.02 * np.abs(d).mean()
        if best is None or inl.sum() > best[1].sum():
            best = (coef, inl)
    coef, inl = best
    coef, *_ = np.linalg.lstsq(A[inl], d[inl], rcond=None)
    # 1/Z = n . (x_n, y_n, 1) / c  with n pointing away from the camera side; we want n "up" (toward camera)
    n_up = -coef / np.linalg.norm(coef)
    tilt_source = "depth_model_plane_zero_shift"
    if camera_pitch_deg is not None:
        # level horizon (no roll) + given downward pitch: world "up" in OpenCV camera axes
        phi = np.deg2rad(camera_pitch_deg)
        n_depth = n_up
        n_up = np.array([0.0, -np.cos(phi), -np.sin(phi)])
        tilt_source = "cli_camera_pitch_level_horizon"
        depth_tilt_disagreement = float(np.rad2deg(np.arccos(np.clip(n_up @ n_depth, -1, 1))))
    k = k_cand if k_cand is not None else int(np.where(np.isfinite(hand_cam_raw[:, 0, 0]))[0][0])
    pinch = 0.5 * (hand_cam_raw[k, 4] + hand_cam_raw[k, 8])
    offset = 0.02 - n_up @ pinch                     # n.P + d0 = 0.02 m above the table
    info = {"fallback_plane_inlier_fraction": round(float(inl.mean()), 3),
            "fallback_reference_frame": int(k), "assumed_pinch_height_m": 0.02,
            "fallback_metric_scale_source": "average_hand", "fallback_tilt_source": tilt_source}
    if camera_pitch_deg is not None:
        info["camera_pitch_deg_cli"] = camera_pitch_deg
        info["depth_model_tilt_disagreement_deg"] = round(depth_tilt_disagreement, 2)
    calib = cal.calibration_from_plane(K, n_up, offset, pinch, focal_source, dict(info))
    bbox0 = track_o.bbox(0)
    if object_size is not None and bbox0 is not None:
        # Known object size beats an average hand: everything metric scales with the
        # camera-to-table distance, so rescale the plane until the frame-0 box fit
        # comes out at the known size.
        from demo_extractor.geometry import fit_box_on_table
        fitted = []
        for yaw in (0.0, np.pi / 4):
            _, size, _ = fit_box_on_table(calib, bbox0, yaw)
            fitted.append(size)
        fitted = np.mean(fitted, axis=0)
        known = np.asarray(object_size, float)
        ratio = float(np.mean([known[0] / fitted[0], known[2] / fitted[2]]))
        info.update({"fallback_metric_scale_source": "known_object_size",
                     "known_size_rescale_factor": round(ratio, 4),
                     "object_size_at_hand_scale_m": np.round(fitted, 4).tolist()})
        calib = cal.calibration_from_plane(K, n_up, offset * ratio, pinch * ratio, focal_source, dict(info))
    return calib


def depth_cross_check(clip, calib, track_o, hand_px, device, depth_model=None, max_frames=40):
    """Depth Anything V2, aligned per frame to the A4-measured table plane, as an
    independent estimate of the carried object's distance from the camera."""
    from demo_extractor.depth import DepthModel, align_to_table, table_depth_map

    T = len(clip.frames)
    areas = np.array([track_o.area(k) for k in range(T)])
    if not areas.any():
        return None, {}
    cent0 = track_o.centroid(0)
    moving = [k for k in range(T) if track_o.centroid(k) is not None and cent0 is not None
              and np.linalg.norm(track_o.centroid(k) - cent0) > 0.5 * np.sqrt(max(areas[0], 1))]
    if not moving:
        return None, {}
    frames = sorted({moving[i] for i in np.linspace(0, len(moving) - 1, min(max_frames, len(moving))).astype(int)})
    own_model = depth_model is None
    if own_model:
        wait_for_memory()
        depth_model = DepthModel(THIRD_PARTY / "depth_anything_v2_small", device=device)
    h, w = clip.frames[0].shape[:2]
    table_z = table_depth_map(calib, (h, w))
    a4 = np.zeros((h, w), np.uint8)
    hw, hh = cal.A4_LONG_M / 2, cal.A4_SHORT_M / 2
    region = calib.project_table(np.array([[-1.6 * hw, -1.6 * hh, 0], [1.6 * hw, -1.6 * hh, 0],
                                           [1.6 * hw, 1.6 * hh, 0], [-1.6 * hw, 1.6 * hh, 0]]))
    cv2.fillConvexPoly(a4, np.round(region).astype(np.int32), 1)
    out, rel_rms = {}, []
    for k in frames:
        disp = depth_model.disparity(clip.frames[k])
        table_mask = a4.astype(bool)
        mask = track_o.masks[k]
        table_mask &= ~cv2.dilate(mask.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
        if np.isfinite(hand_px[k, 0, 0]):
            x0, y0 = np.nanmin(hand_px[k], 0) - 30
            x1, y1 = np.nanmax(hand_px[k], 0) + 30
            table_mask[int(max(y0, 0)):int(y1), int(max(x0, 0)):int(x1)] = False
        a, b, rms = align_to_table(disp, table_z, table_mask)
        if not np.isfinite(a):
            continue
        core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if core.sum() < 10:
            continue
        inv = a * disp[core] + b
        z = 1.0 / np.median(inv[inv > 0]) if np.any(inv > 0) else np.nan
        c = track_o.centroid(k)
        ray = calib.ray_cam(c[None])[0]
        if np.isfinite(z):
            out[k] = float(z / ray[2])
            rel_rms.append(rms)
    if own_model:
        depth_model.close()
    return out, {"depth_model_frames": len(out),
                 "depth_model_table_fit_rel_rms": round(float(np.median(rel_rms)), 4) if rel_rms else None}


def r(x, nd=4):
    return np.round(np.asarray(x, float), nd).tolist()


def build_spec(args, clip, calib, geom, wmap, ws_info, track_o, hand_rms, z_sign, warnings, depth_quality,
               hand_backend="", camera_motion_px=float("nan")):
    obj = geom.object_est
    ev = geom.events
    start_r = wmap.point(obj.start_xyz)[0]
    place_r = wmap.point(geom.place_xyz)[0]
    dense_r = wmap.point(geom.dense_xyz)
    approach_r = wmap.direction(geom.approach_dir)
    open_w = args.gripper_max_open

    def obj_at(t):
        return np.array([np.interp(t, geom.dense_t, dense_r[:, j]) for j in range(3)])

    t_g, t_l, t_s, t_r = ev["grasp_time"], ev["lift_off_time"], ev["set_down_time"], ev["release_time"]
    gw = geom.grasp_width_m
    plan = [
        ("pre_grasp", start_r - 0.08 * approach_r, open_w, max(0.0, t_g - 0.6)),
        ("grasp_open", start_r, open_w, max(0.0, t_g - 0.15)),
        ("grasp_closed", start_r, gw, t_g),
        ("lift", obj_at(t_l + 0.25 * (t_s - t_l)), gw, t_l + 0.25 * (t_s - t_l)),
        ("carry_mid", obj_at(t_l + 0.5 * (t_s - t_l)), gw, t_l + 0.5 * (t_s - t_l)),
        ("pre_place", obj_at(t_l + 0.8 * (t_s - t_l)), gw, t_l + 0.8 * (t_s - t_l)),
        ("place", place_r, gw, t_s),
        ("release", place_r, open_w, max(t_r, t_s + 0.05)),
        ("retreat", place_r + np.array([0, 0, 0.08]), open_w, max(t_r, t_s + 0.05) + 0.5),
    ]
    apex = int(np.argmax(dense_r[:, 2]))
    keyframes = [("start", start_r, t_l), ("apex", dense_r[apex], float(geom.dense_t[apex])),
                 ("place", place_r, t_s)]
    human = {
        "object_start_pose_xyz_yaw": r([*obj.start_xyz, obj.start_yaw]),
        "place_target_xyz": r(geom.place_xyz),
        "place_yaw": round(geom.place_yaw, 4),
        "approach_dir": r(geom.approach_dir),
        "object_waypoints": r(np.column_stack([geom.dense_xyz, geom.dense_t])),
    }
    max_ap = geom.human_pinch_aperture_m
    spec = {
        "schema_version": SCHEMA_VERSION,
        "frame": {"name": "robot_table", "origin": "robot base, on the table surface",
                  "axes": "+x forward, +y left, +z up", "units": "metres, radians, seconds (video time)"},
        "object": {
            "size_m": r(obj.size_m),
            "size_note": "[width along the grasp closing axis, depth, height]; physical, never scaled",
            "size_source": obj.size_source,
            "shape_hint": "box",
            "start_pose_xyz_yaw": r([*start_r, wmap.yaw_angle(obj.start_yaw)]),
            "place_pose_xyz_yaw": r([*place_r, wmap.yaw_angle(geom.place_yaw)]),
            "pose_note": "xyz = object centre; yaw = direction of the grasp closing axis (object faces assumed aligned)",
        },
        "place_target_xyz": r(place_r),
        "object_waypoints": r(np.column_stack([dense_r, geom.dense_t])),
        "object_waypoints_fields": ["x", "y", "z", "t"],
        "object_waypoints_note": f"object centre path, smoothed and resampled at {args.dense_hz:g} Hz from "
                                 "lift-off - 0.5 s to set-down + 0.5 s; static ends held exactly",
        "object_keyframes": [{"label": lab, "xyz": r(p), "t": round(float(t), 4)} for lab, p, t in keyframes],
        "approach_dir": r(approach_r),
        "approach_dir_note": "unit vector the gripper should travel along into the grasp = the human hand's "
                             "wrist->pinch axis at the grasp (use for the pre-grasp offset / approach side)",
        "approach_top_down_fraction": round(float(max(0.0, -approach_r[2])), 3),
        "gripper": {
            "max_opening_m": open_w,
            "grasp_width_m": round(gw, 4),
            "grasp_yaw_rad": round(wmap.yaw_angle(obj.start_yaw), 4),
            "closing_axis": r(wmap.direction(geom.closing_axis)),
            "approach_axis_from_hand": r(wmap.direction(geom.approach_axis_hand)),
            "human_pinch_aperture_m": None if not np.isfinite(max_ap) else round(max_ap, 4),
            "note": "two-finger pinch retarget: width = object width along the thumb-index axis, clamped to "
                    "max_opening; human aperture is tip-to-tip and includes finger thickness",
        },
        "waypoints": [[*r(p), round(float(wd), 4), round(float(t), 4)] for _, p, wd, t in plan],
        "waypoint_fields": ["x", "y", "z", "gripper_width", "t"],
        "waypoint_labels": [lab for lab, *_ in plan],
        "waypoints_note": "SUGGESTED gripper plan derived from the object path (object-centric: gripper centre = "
                          "object centre while held). Not the human hand path.",
        "grasp_time": round(t_g, 4),
        "release_time": round(t_r, 4),
        "lift_off_time": round(t_l, 4),
        "set_down_time": round(t_s, 4),
        "phases": [{**ph, "t_start": round(ph["t_start"], 4), "t_end": round(ph["t_end"], 4)} for ph in geom.phases],
        "workspace_mapping": {**wmap.as_dict(), **ws_info},
        "human_table_frame": {"definition": "origin = A4 sheet centre on the table; +x along the sheet's long edge "
                                            "(pointing right in the image); +z up" if calib.method == "a4" else
                              "no-A4 fallback: origin under the hand at grasp; +x = camera right projected on the table",
                              **human},
        "source_video": {"path": str(Path(clip.path).resolve()), "source_fps": round(clip.source_fps, 3),
                         "source_resolution": list(clip.source_size), "source_frames": clip.source_frame_count,
                         "processed_fps": round(clip.processed_fps, 3), "processed_resolution": list(clip.size),
                         "processed_frames": len(clip.frames), "processing_scale": round(clip.scale, 5)},
        "calibration": {"method": calib.method, "focal_source": calib.focal_source,
                        "K_processing_px": r(calib.K, 3),
                        "R_camera_from_table": r(calib.R_ct, 6), "t_camera_from_table_m": r(calib.t_ct, 5),
                        "hand_scale_factor": round(geom.hand_scale, 4),
                        "hand_scale_source": geom.hand_scale_source,
                        "mediapipe_world_z_sign": z_sign,
                        "hand_backend": hand_backend,
                        **{k: v for k, v in calib.info.items()}},
        "quality": {
            **geom.quality, **depth_quality,
            "camera_motion_px_first_to_last": round(camera_motion_px, 2),
            "hand_lift_reprojection_rms_px_median": round(float(np.nanmedian(hand_rms)), 3)
            if np.isfinite(hand_rms).any() else None,
            "events": {k: v for k, v in ev.items() if not k.endswith("_time")},
            "object_prompt": {k: v for k, v in track_o.prompt.items() if k not in ("points", "labels", "box")},
            "warnings": warnings,
            "real_video_validated": False,
        },
        "progress_curve": None,
        "files": {"trajectory_debug": "trajectory_debug.json", "overlay": "debug_overlay.mp4",
                  "plot": "trajectories_3d.png", "object_masks": "object_masks.npz"},
    }
    return spec


def write_debug(path, clip, geom, track_o, hand_ok) -> None:
    frames = []
    rej = {k: reason for reason, ks in geom.rejected.items() for k in ks}
    for k, t in enumerate(clip.times):
        frames.append({
            "t": round(float(t), 4),
            "object_xyz_table": None if not np.isfinite(geom.object_xyz_raw[k, 0]) else r(geom.object_xyz_raw[k]),
            "object_rejected_reason": rej.get(k),
            "object_mask_area_px": track_o.area(k),
            "object_score_logit": round(float(track_o.scores[k]), 3),
            "hand_detected": bool(hand_ok[k]),
            "hand_pinch_xyz_table_debug_only": None if not np.isfinite(geom.pinch_table[k, 0]) else r(geom.pinch_table[k]),
            "pinch_aperture_m": None if not np.isfinite(geom.aperture_m[k]) else round(float(geom.aperture_m[k]), 4),
            "object_ray_distance_sources_m": geom.depth_sources.get(k),
        })
    path.write_text(json.dumps({"frames": frames, "dense_object_path_table": r(
        np.column_stack([geom.dense_xyz, geom.dense_t]))}, indent=1))


if __name__ == "__main__":
    main()
