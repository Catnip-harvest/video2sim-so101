"""From 2D tracks to a metric, object-centric description of the demo.

The object's path is the primary output (X-Sim style): a human arm hangs from
the shoulder and the SO-101 is table-mounted, so the human hand path does not
transfer; the object path does. The hand is used only for
  * timing   - grasp (thumb-index closes) and release (they open again),
  * approach - which way the hand came in,
  * width    - how wide the pinch was,
  * scale    - one number: the person's true hand size, calibrated at the grasp
               contact, which also gives the carried object's depth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from .hands import INDEX_TIP, MIDDLE_TIP, THUMB_TIP, WRIST


@dataclass
class ObjectEstimate:
    size_m: np.ndarray              # (width along closing axis, depth, height)
    start_xyz: np.ndarray           # centre, table frame
    start_yaw: float
    size_source: str
    fit_rms_px: float


@dataclass
class DemoGeometry:
    times: np.ndarray
    object_est: ObjectEstimate
    events: dict                    # grasp/lift/set/release times and frame indices
    hand_scale: float
    hand_scale_source: str
    pinch_table: np.ndarray         # (T,3) k-scaled pinch midpoint in table frame (debug only)
    aperture_m: np.ndarray          # (T,) k-scaled thumb-index tip distance
    object_xyz_raw: np.ndarray      # (T,3) per-frame object centre measurements (NaN = rejected)
    rejected: dict                  # reason -> list of frame indices
    dense_t: np.ndarray             # dense, smoothed object path (table frame)
    dense_xyz: np.ndarray
    place_xyz: np.ndarray
    place_yaw: float
    approach_dir: np.ndarray        # unit vector, table frame, hand motion into the grasp
    approach_axis_hand: np.ndarray  # unit vector, wrist -> pinch at grasp
    closing_axis: np.ndarray        # unit vector, thumb -> index at grasp
    grasp_width_m: float
    human_pinch_aperture_m: float
    phases: list[dict]
    warnings: list[str] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    depth_sources: dict = field(default_factory=dict)


# --------------------------------------------------------------- box fitting
def box_corners(center_xy, z_base, size, yaw) -> np.ndarray:
    w, d, h = size
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[sx * w / 2, sy * d / 2, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (0.0, h)])
    xy = local[:, :2] @ np.array([[c, s], [-s, c]])
    return np.column_stack([xy + center_xy, local[:, 2] + z_base])


def projected_bbox(calib, corners) -> np.ndarray:
    uv = calib.project_table(corners)
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def fit_box_on_table(calib, bbox_px: np.ndarray, yaw: float, size_fixed=None, z_base: float = 0.0):
    """Upright box resting at z_base whose projection matches the mask's bbox.
    Unknowns: centre x, y, and (unless given) footprint width and height."""
    bottom_mid = np.array([[0.5 * (bbox_px[0] + bbox_px[2]), bbox_px[3]]])
    guess_xy = calib.intersect_table_plane(bottom_mid, z_base)[0, :2]
    depth = np.linalg.norm(calib.table_to_cam(np.array([[*guess_xy, z_base]]))[0])
    w_guess = max(0.01, 0.8 * (bbox_px[2] - bbox_px[0]) * depth / calib.K[0, 0])
    away = guess_xy - calib.camera_center_table()[:2]
    away /= np.linalg.norm(away) + 1e-9
    guess_xy = guess_xy + 0.5 * w_guess * away

    if size_fixed is not None:
        size = np.asarray(size_fixed, float)

        def res(p):
            return projected_bbox(calib, box_corners(p[:2], z_base, size, yaw)) - bbox_px
        sol = least_squares(res, guess_xy, method="lm")
        return sol.x, size, float(np.sqrt(np.mean(sol.fun ** 2)))

    def res(p):
        x, y, w, h = p
        return projected_bbox(calib, box_corners((x, y), z_base, (w, w, h), yaw)) - bbox_px

    sol = least_squares(res, [*guess_xy, w_guess, w_guess], bounds=([-5, -5, 0.005, 0.005], [5, 5, 0.3, 0.4]))
    x, y, w, h = sol.x
    return np.array([x, y]), np.array([w, w, h]), float(np.sqrt(np.mean(sol.fun ** 2)))


# ------------------------------------------------------------------ helpers
def _first_run(cond: np.ndarray, length: int, start: int = 0) -> int | None:
    count = 0
    for k in range(start, len(cond)):
        count = count + 1 if cond[k] else 0
        if count >= length:
            return k - length + 1
    return None


def _nan_interp(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = y.copy()
    for j in range(y.shape[1]):
        ok = np.isfinite(y[:, j])
        if ok.sum() >= 2:
            out[:, j] = np.interp(t, t[ok], y[ok, j])
    return out


def _mask_bottom(mask) -> np.ndarray | None:
    if mask is None or mask.sum() < 5:
        return None
    ys, xs = np.nonzero(mask)
    y_max = ys.max()
    return np.array([np.median(xs[ys >= y_max - 2]), float(y_max)])


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ------------------------------------------------------------------ pipeline
def analyse_demo(times, calib, hand_px, hand_cam_raw, track, image_size, *, object_size=None,
                 pinch_mode="thumb-index", gripper_max_open=0.06, dense_hz=30.0,
                 depth_cross_check=None) -> DemoGeometry:
    T = len(times)
    w_img, h_img = image_size
    warnings: list[str] = []
    quality: dict = {}
    dt = float(np.median(np.diff(times))) if T > 1 else 1 / 15

    # ---- hand quantities (scale-free) --------------------------------------
    hand_ok = np.isfinite(hand_cam_raw[:, 0, 0])
    if pinch_mode == "thumb-index-middle":
        finger_tip_raw = 0.5 * (hand_cam_raw[:, INDEX_TIP] + hand_cam_raw[:, MIDDLE_TIP])
        finger_tip_px = 0.5 * (hand_px[:, INDEX_TIP] + hand_px[:, MIDDLE_TIP])
    else:
        finger_tip_raw = hand_cam_raw[:, INDEX_TIP]
        finger_tip_px = hand_px[:, INDEX_TIP]
    pinch_raw = 0.5 * (hand_cam_raw[:, THUMB_TIP] + finger_tip_raw)
    pinch_px = 0.5 * (hand_px[:, THUMB_TIP] + finger_tip_px)
    aperture_raw = np.linalg.norm(hand_cam_raw[:, THUMB_TIP] - finger_tip_raw, axis=1)
    closing_cam = finger_tip_raw - hand_cam_raw[:, THUMB_TIP]

    # ---- object observations in 2D -----------------------------------------
    centroids = np.array([c if (c := track.centroid(k)) is not None else [np.nan, np.nan] for k in range(T)])
    areas = np.array([track.area(k) for k in range(T)], float)
    present = np.isfinite(centroids[:, 0]) & (track.scores > 0)

    def hand_far(k: int, bbox) -> bool:
        if not hand_ok[k] or bbox is None:
            return True
        diag = np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
        nearest = np.min(np.linalg.norm(hand_px[k] - 0.5 * (bbox[:2] + bbox[2:]), axis=1))
        return nearest > 1.2 * diag + 10

    start_frames = [k for k in range(min(T, int(2.0 / dt) + 1)) if present[k] and hand_far(k, track.bbox(k))]
    if not start_frames:
        start_frames = [k for k in range(T) if present[k]][:3]
    if not start_frames:
        raise RuntimeError("object never tracked; check the object prompt")
    start_frames = start_frames[:10]
    area0 = float(np.median(areas[start_frames]))
    c_start = np.median(centroids[start_frames], axis=0)
    bbox_start = np.median(np.array([track.bbox(k) for k in start_frames]), axis=0)
    size_px = np.sqrt(area0)

    # ---- lift-off and set-down from object motion ---------------------------
    # Motion signal robust to finger occlusion: the smaller of the centroid
    # shift and the mask-bottom shift (fingers hide the sides/top of a pinched
    # object, rarely its bottom edge; a real move shifts both).
    bottoms = np.array([b if (b := _mask_bottom(track.masks[k])) is not None else [np.nan, np.nan]
                        for k in range(T)])
    b_start = np.median(bottoms[start_frames], axis=0)
    thr = max(4.0, 0.5 * size_px)
    disp0 = np.minimum(np.linalg.norm(centroids - c_start, axis=1), np.linalg.norm(bottoms - b_start, axis=1))
    moving = present & (disp0 > thr)
    k_cross = _first_run(moving, 3)
    if k_cross is None:
        raise RuntimeError("the object never moved in the video: no pick detected")
    k_lift = k_cross
    while k_lift > 0 and k_cross - k_lift < int(1.0 / dt) and not (present[k_lift] and disp0[k_lift] <= 0.35 * thr):
        k_lift -= 1

    later = [k for k in range(T - 1, k_cross, -1) if present[k]]
    end_frames = []
    gone_at_end = not later or (T - 1 - later[0]) * dt > 0.5   # object absent for the last 0.5 s
    for k in ([] if gone_at_end else later):   # last frames where the object is visible, static, hand away
        if hand_far(k, track.bbox(k)):
            end_frames.append(k)
        if len(end_frames) >= 8:
            break
    object_left_view = False
    if end_frames:
        c_end = np.median(centroids[end_frames], axis=0)
        b_end = np.median(bottoms[end_frames], axis=0)
        disp_end = np.minimum(np.linalg.norm(centroids - c_end, axis=1), np.linalg.norm(bottoms - b_end, axis=1))
        moved_from_end = present & (disp_end > thr)
        moved_from_end[:k_cross] = True
        k_set_cross = int(np.max(np.where(moved_from_end)[0]))
        k_set = k_set_cross + 1
        while k_set < T - 1 and not (present[k_set] and disp_end[k_set] <= 0.35 * thr):
            k_set += 1
        spread = np.linalg.norm(centroids[max(end_frames)] - centroids[min(end_frames)])
        if spread > thr:
            warnings.append("object still moving at the end of the clip; place pose unreliable")
    else:
        object_left_view = True
        k_set = max(k for k in range(T) if present[k])
        warnings.append("object left the view (or was occluded) before being set down; "
                        "place = last tracked position")
    k_set = int(min(max(k_set, k_lift + 1), T - 1))

    # ---- grasp and release from the hand aperture ---------------------------
    ap = aperture_raw.copy()
    events = {"lift_off_frame": int(k_lift), "set_down_frame": int(k_set)}
    win_lo = max(0, k_lift - int(2.0 / dt))
    window = [k for k in range(win_lo, min(T, k_lift + 2)) if hand_ok[k]]
    if len(window) >= 3:
        k_min = window[int(np.argmin(ap[window]))]
        before = [k for k in window if k <= k_min]
        a_open = float(np.max(ap[before]))
        a_min = float(ap[k_min])
        start_idx = before[int(np.argmax(ap[before]))]
        k_grasp = next(k for k in window if k >= start_idx and ap[k] <= a_min + 0.15 * (a_open - a_min))
        grasp_source = "hand_aperture"
        if a_open - a_min < 0.01:
            warnings.append("thumb-index gap barely changed before lift-off (<1 cm): grasp time is weak")
        k_grasp = min(k_grasp, k_lift)          # the fingers close before the object moves
    else:
        k_grasp = max(0, k_lift - int(0.2 / dt))
        a_open = a_min = np.nan
        grasp_source = "object_motion_only"
        warnings.append("hand not detected around the grasp; grasp time = lift-off - 0.2 s")
    hold = [k for k in range(k_grasp, k_set + 1) if hand_ok[k]]
    a_closed = float(np.median(ap[hold])) if hold else np.nan
    release_source = "object_motion_only"
    k_release = min(T - 1, k_set + int(0.1 / dt))
    if hold and np.isfinite(a_open) and not object_left_view:
        for k in range(max(k_grasp + 1, k_set - int(0.3 / dt)), min(T, k_set + int(2.0 / dt))):
            if hand_ok[k] and ap[k] > a_closed + 0.3 * (a_open - a_closed):
                # back up to where the fingers started to open
                opening_started = a_closed + 0.08 * (a_open - a_closed)
                while k - 1 > k_set - int(0.3 / dt) and hand_ok[k - 1] and ap[k - 1] > opening_started:
                    k -= 1
                k_release, release_source = k, "hand_aperture"
                break
            if not hand_ok[k] and k > k_set:   # hand vanished right after set-down
                k_release, release_source = k, "hand_lost_after_set_down"
                break
    events.update({"grasp_frame": int(k_grasp), "release_frame": int(k_release),
                   "grasp_source": grasp_source, "release_source": release_source})

    # ---- grasp yaw (closing axis) and the object's start pose ---------------
    grasp_frames = [k for k in range(k_grasp, k_lift + 1) if hand_ok[k]] or \
                   [k for k in range(max(0, k_grasp - 2), min(T, k_grasp + 3)) if hand_ok[k]]
    if grasp_frames:
        closing_tab = np.median(closing_cam[grasp_frames] @ calib.R_ct, axis=0)
        closing_axis = _unit(closing_tab)
        grasp_yaw = float(np.arctan2(closing_axis[1], closing_axis[0]))
    else:
        closing_axis, grasp_yaw = np.array([1.0, 0, 0]), 0.0
        warnings.append("no hand at grasp: object yaw unknown (0 used)")

    center_xy, size, fit_rms = fit_box_on_table(calib, bbox_start, grasp_yaw, size_fixed=object_size)
    obj = ObjectEstimate(size_m=size, start_xyz=np.array([*center_xy, size[2] / 2]), start_yaw=grasp_yaw,
                         size_source="cli" if object_size is not None else "mask_box_fit", fit_rms_px=fit_rms)
    quality["object_box_fit_rms_px"] = round(fit_rms, 3)

    # ---- hand size: one scale factor from the grasp contact -----------------
    start_cam = calib.table_to_cam(obj.start_xyz[None])[0]
    if grasp_frames:
        ratios = [np.linalg.norm(start_cam) / np.linalg.norm(pinch_raw[k]) for k in grasp_frames]
        hand_scale = float(np.median(ratios))
        hand_scale_source = "grasp_contact"
        if not 0.7 <= hand_scale <= 1.4:
            warnings.append(f"hand scale {hand_scale:.2f} outside 0.7-1.4: unusual hand or poor monocular hand "
                            "depth (hand depth has the lowest weight in the object-depth fusion)")
    else:
        hand_scale, hand_scale_source = 1.0, "average_hand_prior"
    pinch_cam = hand_scale * pinch_raw
    pinch_table = calib.cam_to_table(pinch_cam.reshape(-1, 3)).reshape(-1, 3)
    aperture_m = hand_scale * aperture_raw
    grasp_pinch_dist = np.linalg.norm(pinch_cam[grasp_frames], axis=1).mean() if grasp_frames else np.nan
    depth_offset = float(np.linalg.norm(start_cam) - grasp_pinch_dist) if grasp_frames else 0.0
    cam_tab = calib.camera_center_table()
    centroid_offset_px = calib.project_table(obj.start_xyz[None])[0] - c_start
    visible_bias = 0.3 * float(obj.size_m[0])   # depth-model median sees the front faces, not the centre

    def fuse_distance(k: int, ray_tab: np.ndarray, plane=None, table_prior=False) -> tuple[float, dict]:
        """Distance along the object's view ray, fused from independent sources
        (inverse-variance weights; sigmas are rough, documented in README)."""
        est = {}
        if plane is not None:
            point, normal = plane
            denom = float(ray_tab @ normal)
            if abs(denom) > 0.1:
                d = float((point - cam_tab) @ normal / denom)
                if d > 0:
                    est["transport_plane"] = (d, 0.02 / abs(denom))
        if depth_cross_check is not None and k in depth_cross_check:
            d = depth_cross_check[k] + visible_bias
            est["depth_model"] = (d, 0.04 * d)
        if hand_ok[k] and grasp_frames:
            d = float(np.linalg.norm(pinch_cam[k]) + depth_offset)
            est["hand"] = (d, 0.15 * d)
        if table_prior and abs(ray_tab[2]) > 0.05:
            d = float((obj.size_m[2] / 2 - cam_tab[2]) / ray_tab[2])
            if d > 0:
                est["rests_on_table"] = (d, 0.02 / abs(ray_tab[2]))
        if not est:
            return np.nan, {}

        def combine(items):
            wsum = sum(1 / s ** 2 for _, s in items)
            return sum(d / s ** 2 for d, s in items) / wsum

        fused = combine(est.values())
        if len(est) >= 3:   # drop the one source farthest (in its own sigmas) from the median
            med = float(np.median([d for d, _ in est.values()]))
            z = {name: abs(d - med) / sig for name, (d, sig) in est.items()}
            worst = max(z, key=z.get)
            if z[worst] > 3:
                del est[worst]
            fused = combine(est.values())
        depth_sources[k] = {name: round(d, 4) for name, (d, _) in est.items()} | {"fused": round(float(fused), 4)}
        return float(fused), {name: d for name, (d, _) in est.items()}

    disagreement: dict[str, list] = {}
    depth_sources: dict[int, dict] = {}

    def note(sources: dict, fused: float) -> None:
        for name, d in sources.items():
            disagreement.setdefault(name, []).append(abs(d - fused) / fused)

    # ---- place pose ----------------------------------------------------------
    # Direction from the object's own final mask once the hand is gone; distance
    # fused from the depth model, the hand at release, and "rests near table height".
    rel_frames = [k for k in range(max(k_set - 1, 0), min(T, k_release + 1)) if hand_ok[k]]
    raw = np.full((T, 3), np.nan)
    rejected = {"no_mask": [], "occluded": [], "mask_leak": [], "fast_motion": [], "no_depth": []}
    if object_left_view:
        place = None
        place_source = "last_tracked_position"
    else:
        c_end_px = np.median(centroids[end_frames], axis=0) + centroid_offset_px
        ray_end = calib.ray_cam(c_end_px[None])[0] @ calib.R_ct
        dists, rich = [], []
        for k in sorted(set(end_frames) | set(rel_frames)):
            d, src = fuse_distance(k, ray_end, table_prior=True)
            if np.isfinite(d):
                dists.append(d)
                if "depth_model" in src:   # frames with the strongest measurement available
                    rich.append(d)
                note({f"place_{n}": v for n, v in src.items()}, d)
        if dists:
            place = cam_tab + float(np.median(rich if rich else dists)) * ray_end
            place_source = "final_mask_ray_fused_depth"
        else:
            place = calib.intersect_table_plane(c_end_px[None], obj.size_m[2] / 2)[0]
            place_source = "final_mask_ray_on_table_plane"
            warnings.append("no depth source at the place: assumed to rest on the table plane")

    # ---- per-frame object centre during transport ---------------------------
    speed_px = np.r_[0.0, np.linalg.norm(np.diff(centroids, axis=0), axis=1)]
    plane = None
    if place is not None and np.linalg.norm((place - obj.start_xyz)[:2]) > 0.03:
        n_plane = _unit(np.cross(place - obj.start_xyz, [0.0, 0.0, 1.0]))
        plane = (obj.start_xyz, n_plane)
    for k in range(k_lift + 1, k_set):
        if not present[k]:
            rejected["no_mask"].append(k)
            continue
        if areas[k] < 0.4 * area0:
            rejected["occluded"].append(k)
            continue
        if areas[k] > 2.5 * area0:
            rejected["mask_leak"].append(k)
            continue
        if np.isfinite(speed_px[k]) and speed_px[k] > 1.0 * size_px:
            rejected["fast_motion"].append(k)
            continue
        ray_tab = calib.ray_cam((centroids[k] + centroid_offset_px)[None])[0] @ calib.R_ct
        dist, src = fuse_distance(k, ray_tab, plane=plane)
        if not np.isfinite(dist):
            rejected["no_depth"].append(k)
            continue
        note(src, dist)
        raw[k] = cam_tab + dist * ray_tab
    raw[k_lift] = obj.start_xyz
    if place is None:
        place = raw[np.isfinite(raw[:, 0])][-1] if np.isfinite(raw[:, 0]).any() else obj.start_xyz
    raw[k_set] = place
    quality["depth_source_median_rel_disagreement"] = {n: round(float(np.median(v)), 4)
                                                       for n, v in disagreement.items()}
    quality["transport_plane_prior_used"] = plane is not None
    if place[2] < obj.size_m[2] / 2 - 0.02:
        warnings.append(f"place height {place[2]:.3f} m is below the table; depth at release is off")
    yaw_frames = [k for k in rel_frames if np.linalg.norm(closing_cam[k]) > 0]
    place_yaw = grasp_yaw
    if yaw_frames:
        c_rel = _unit(np.median(closing_cam[yaw_frames] @ calib.R_ct, axis=0))
        place_yaw = float(np.arctan2(c_rel[1], c_rel[0]))

    # ---- continuity: the carried path must start/end exactly at the rest poses
    seg = np.arange(k_lift, k_set + 1)
    ok = np.isfinite(raw[seg, 0])
    if ok.sum() >= 2:
        seg_t = times[seg]
        first, last = seg[ok][0], seg[ok][-1]
        off_start = obj.start_xyz - raw[first]
        off_end = place - raw[last]
        u = np.clip((seg_t - times[first]) / max(times[last] - times[first], 1e-6), 0, 1)[:, None]
        raw[seg] = np.where(ok[:, None], raw[seg] + (1 - u) * off_start + u * off_end, np.nan)

    # ---- smooth + densely interpolate (static ends are held exactly) --------
    path = np.full((T, 3), np.nan)
    path[:k_lift + 1] = obj.start_xyz
    path[k_set:] = place
    path[k_lift:k_set + 1] = raw[k_lift:k_set + 1]
    path = _nan_interp(times, path)
    inner = slice(k_lift, k_set + 1)
    n_inner = k_set - k_lift + 1
    win = int(round(0.35 / dt)) | 1
    if n_inner >= 5:
        win = min(win, n_inner if n_inner % 2 == 1 else n_inner - 1)
        if win >= 5:
            smoothed = savgol_filter(path[inner], win, 2, axis=0)
            smoothed[0], smoothed[-1] = obj.start_xyz, place
            path[inner] = smoothed
    path[:, 2] = np.maximum(path[:, 2], min(obj.size_m[2] / 2, place[2]) - 0.005)
    t0 = max(times[0], times[k_lift] - 0.5)
    t1 = min(times[-1], times[k_set] + 0.5)
    dense_t = np.arange(t0, t1 + 1e-9, 1.0 / dense_hz)
    dense_xyz = PchipInterpolator(times, path, axis=0)(dense_t)

    # ---- approach -------------------------------------------------------------
    # Primary: the hand's own axis at the grasp (wrist -> pinch). It comes from the
    # shape of the hand, not its distance from the camera, so monocular depth noise
    # (+-15% per frame) cancels. Secondary, for reference only: the direction the
    # pinch point travelled over its last ~8 cm (depth noise makes this rough).
    approach_motion_dir = None
    if hand_ok[k_grasp]:
        p_grasp = pinch_table[k_grasp]
        for k in range(k_grasp - 1, max(-1, k_grasp - int(3.0 / dt)), -1):
            if hand_ok[k] and np.linalg.norm(pinch_table[k] - p_grasp) > 0.08:
                approach_motion_dir = _unit(p_grasp - pinch_table[k])
                break
    if grasp_frames:
        wrist_tab = calib.cam_to_table(hand_scale * hand_cam_raw[grasp_frames, WRIST])
        approach_axis_hand = _unit(np.median(pinch_table[grasp_frames] - wrist_tab, axis=0))
        approach_dir = approach_axis_hand
    else:
        approach_axis_hand = approach_dir = np.array([0.0, 0.0, -1.0])
        warnings.append("no hand at the grasp: approach_dir defaulted to top-down")
    quality["approach_motion_dir_table"] = (None if approach_motion_dir is None
                                            else np.round(approach_motion_dir, 3).tolist())

    grasp_width = float(min(obj.size_m[0], gripper_max_open))
    if obj.size_m[0] > gripper_max_open:
        warnings.append(f"object width {obj.size_m[0]:.3f} m exceeds gripper max opening {gripper_max_open} m")
    human_ap = float(np.median(aperture_m[grasp_frames])) if grasp_frames else float("nan")

    t_grasp, t_lift, t_set, t_release = (float(times[i]) for i in (k_grasp, k_lift, k_set, k_release))
    t_end = float(times[-1])
    phases = [
        {"name": "approach", "t_start": float(times[0]), "t_end": t_grasp},
        {"name": "grasp", "t_start": t_grasp, "t_end": max(t_grasp, t_lift)},
        {"name": "transport", "t_start": max(t_grasp, t_lift), "t_end": t_set},
        {"name": "release", "t_start": t_set, "t_end": min(t_end, max(t_set, t_release) + 0.3)},
        {"name": "retreat", "t_start": min(t_end, max(t_set, t_release) + 0.3), "t_end": t_end},
    ]
    events.update({"grasp_time": t_grasp, "lift_off_time": t_lift, "set_down_time": t_set,
                   "release_time": t_release, "object_left_view": object_left_view,
                   "place_source": place_source})

    n_transport = n_inner
    n_rejected = sum(len(v) for v in rejected.values())
    longest_gap = 0
    gap = 0
    for k in range(k_lift, k_set + 1):
        gap = gap + 1 if not np.isfinite(raw[k, 0]) else 0
        longest_gap = max(longest_gap, gap)
    if longest_gap * dt > 0.5:
        warnings.append(f"object path has a {longest_gap * dt:.2f} s gap filled by interpolation")
    quality.update({
        "transport_frames": int(n_transport),
        "transport_frames_rejected": int(n_rejected),
        "transport_rejected_by_reason": {k: len(v) for k, v in rejected.items()},
        "longest_interpolated_gap_s": round(longest_gap * dt, 3),
        "hand_detected_fraction": round(float(hand_ok.mean()), 3),
        "object_tracked_fraction": round(float(present.mean()), 3),
        "object_area_px_at_rest": int(area0),
        "pick_hand_vs_object_m": round(float(np.linalg.norm(
            np.median(pinch_table[grasp_frames], axis=0) - obj.start_xyz)), 4) if grasp_frames else None,
    })
    return DemoGeometry(
        times=times, object_est=obj, events=events, hand_scale=hand_scale, hand_scale_source=hand_scale_source,
        pinch_table=pinch_table, aperture_m=aperture_m, object_xyz_raw=raw, rejected=rejected,
        dense_t=dense_t, dense_xyz=dense_xyz, place_xyz=place, place_yaw=place_yaw, approach_dir=approach_dir,
        approach_axis_hand=approach_axis_hand, closing_axis=closing_axis, grasp_width_m=grasp_width,
        human_pinch_aperture_m=human_ap, phases=phases, warnings=warnings, quality=quality,
        depth_sources=depth_sources)
