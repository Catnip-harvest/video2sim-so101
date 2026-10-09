"""Camera intrinsics + table frame from an A4 sheet lying on the table.

Plain-words version: a sheet of A4 paper is a rectangle of known size
(297 x 210 mm). Its four corners in the image pin down (a) the camera's focal
length, (b) where the table plane is, (c) the metric scale. Official terms:
plane homography, image of the absolute conic (focal from two orthogonality
constraints), planar PnP (IPPE).

Table frame (the "human table frame"):
  origin  = centre of the A4 sheet, on the table surface
  +x      = along the sheet's long edge, the direction that points right in the image
  +z      = table normal, pointing up (toward the camera side)
  +y      = z cross x
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

A4_LONG_M, A4_SHORT_M = 0.297, 0.210


@dataclass
class Calibration:
    K: np.ndarray                 # 3x3 intrinsics at processing resolution
    R_ct: np.ndarray              # camera-from-table rotation: X_cam = R_ct X_tab + t_ct
    t_ct: np.ndarray
    method: str                   # "a4" or "no_a4_fallback"
    focal_source: str             # "a4_homography", "hfov_prior", "cli"
    info: dict = field(default_factory=dict)

    def table_to_cam(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.R_ct.T + self.t_ct

    def cam_to_table(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self.t_ct) @ self.R_ct

    def project_table(self, pts: np.ndarray) -> np.ndarray:
        cam = self.table_to_cam(np.atleast_2d(pts))
        uv = cam @ self.K.T
        return uv[:, :2] / uv[:, 2:3]

    def ray_cam(self, px: np.ndarray) -> np.ndarray:
        px = np.atleast_2d(px)
        rays = np.stack([(px[:, 0] - self.K[0, 2]) / self.K[0, 0],
                         (px[:, 1] - self.K[1, 2]) / self.K[1, 1], np.ones(len(px))], -1)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def intersect_table_plane(self, px: np.ndarray, height: float = 0.0) -> np.ndarray:
        """Table-frame point where the pixel's ray crosses the plane z = height."""
        rays_tab = self.ray_cam(px) @ self.R_ct        # direction in table frame
        origin_tab = self.camera_center_table()
        s = (height - origin_tab[2]) / rays_tab[:, 2]
        return origin_tab + s[:, None] * rays_tab

    def camera_center_table(self) -> np.ndarray:
        return -self.R_ct.T @ self.t_ct


# ------------------------------------------------------------------ detection
def _order_quad(quad: np.ndarray) -> np.ndarray:
    center = quad.mean(0)
    ang = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    quad = quad[np.argsort(ang)]
    start = int(np.argmin(quad.sum(1)))
    return np.roll(quad, -start, axis=0)


def _refine_corners(contour: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Fit a line to the contour points along each side, intersect neighbours."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        length = np.linalg.norm(ab)
        u = ((pts - a) @ ab) / (length ** 2)
        rel = pts - a
        dist = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / length
        sel = pts[(u > 0.12) & (u < 0.88) & (dist < max(3.0, 0.02 * length))]
        if len(sel) < 6:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    refined = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return quad
        s = np.linalg.solve(A, p2 - p1)
        refined.append(p1 + s[0] * d1)
    refined = np.array(refined)
    if np.max(np.linalg.norm(refined - quad, axis=1)) > 0.05 * np.linalg.norm(quad[0] - quad[2]):
        return quad
    return refined


def detect_a4_quad(frame: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Find the brightest low-saturation convex quadrilateral (the paper)."""
    h, w = frame.shape[:2]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.int16)
    lightness = lab[..., 0].astype(np.uint8)
    chroma = np.abs(lab[..., 1] - 128) + np.abs(lab[..., 2] - 128)
    otsu, _ = cv2.threshold(cv2.GaussianBlur(lightness, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds = sorted({int(otsu), *[int(np.percentile(lightness, p)) for p in (60, 75, 85, 92)]})
    best, best_score = None, 0.0
    for thr in thresholds:
        mask = ((lightness >= thr) & (chroma < 28)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.002 * w * h or area > 0.6 * w * h:
                continue
            hull = cv2.convexHull(contour)
            approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
            if len(approx) != 4 or not cv2.isContourConvex(approx.astype(np.int32)):
                continue
            quad_area = cv2.contourArea(approx.astype(np.float32))
            solidity = area / max(quad_area, 1.0)
            if solidity < 0.85:
                continue
            touches_border = np.any(approx[:, 0] <= 2) or np.any(approx[:, 0] >= w - 3) or \
                np.any(approx[:, 1] <= 2) or np.any(approx[:, 1] >= h - 3)
            if touches_border:
                continue
            inside = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(inside, approx.astype(np.int32), 1)
            brightness = float(lightness[inside > 0].mean()) / 255.0
            score = quad_area * solidity ** 4 * brightness
            if score > best_score:
                best_score = score
                best = _refine_corners(contour, _order_quad(approx.astype(np.float64)))
    return best, best_score


def detect_a4_over_frames(frames: list[np.ndarray], max_frames: int = 12) -> tuple[np.ndarray | None, dict]:
    quads = []
    for frame in frames[:max_frames]:
        quad, _ = detect_a4_quad(frame)
        if quad is not None:
            quads.append(_order_quad(quad))
    if not quads:
        return None, {"a4_detected_frames": 0}
    # keep the quads that agree with the median one (the hand may enter later)
    quads = np.array(quads)
    median = np.median(quads, axis=0)
    spread = np.linalg.norm(quads - median, axis=2).max(1)
    agreeing = quads[spread < max(4.0, 3 * np.median(spread))]
    corners = np.median(agreeing, axis=0)
    jitter = float(np.linalg.norm(agreeing - corners, axis=2).mean()) if len(agreeing) > 1 else float("nan")
    return corners, {"a4_detected_frames": int(len(quads)), "a4_frames_used": int(len(agreeing)),
                     "a4_corner_jitter_px": round(jitter, 3)}


# ---------------------------------------------------------------- calibration
def _model_points(long_first: bool) -> np.ndarray:
    l1, l2 = (A4_LONG_M, A4_SHORT_M) if long_first else (A4_SHORT_M, A4_LONG_M)
    return np.array([[-l1 / 2, -l2 / 2], [l1 / 2, -l2 / 2], [l1 / 2, l2 / 2], [-l1 / 2, l2 / 2]])


def _focal_from_homography(corners_centered: np.ndarray, long_first: bool) -> tuple[float | None, float]:
    """Closed-form focal from the rectangle: returns (f or None, disagreement)
    where disagreement = |log(f_orthogonality / f_equal_length)|."""
    H = cv2.getPerspectiveTransform(_model_points(long_first).astype(np.float32),
                                    corners_centered.astype(np.float32)).astype(np.float64)
    H /= np.linalg.norm(H)
    h1, h2 = H[:, 0], H[:, 1]
    a1, b1 = h1[0] * h2[0] + h1[1] * h2[1], h1[2] * h2[2]
    a2, b2 = h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2, h1[2] ** 2 - h2[2] ** 2
    s = -(a1 * b1 + a2 * b2) / (a1 * a1 + a2 * a2)
    if not np.isfinite(s) or s <= 0:
        return None, np.inf
    single = [-b / a for a, b in ((a1, b1), (a2, b2)) if abs(a) > 1e-9 and -b / a > 0]
    disagreement = abs(0.5 * np.log(single[0] / single[1])) if len(single) == 2 else np.inf
    return 1.0 / np.sqrt(s), disagreement


def _aspect_with_focal(corners_centered: np.ndarray, f: float) -> float:
    """Long/short ratio of the rectangle implied by focal f (c0->c1 over c0->c3)."""
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    H = cv2.getPerspectiveTransform(unit, corners_centered.astype(np.float32)).astype(np.float64)
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    return float(np.linalg.norm(Kinv @ H[:, 0]) / np.linalg.norm(Kinv @ H[:, 1]))


def calibrate_from_a4(corners_px: np.ndarray, image_size: tuple[int, int], hfov_prior_deg: float = 65.0,
                      focal_px_override: float | None = None) -> Calibration:
    w, h = image_size
    pp = np.array([w / 2, h / 2])
    centered = corners_px - pp
    f_prior = (w / 2) / np.tan(np.deg2rad(hfov_prior_deg) / 2)

    info: dict = {}
    options = []
    for long_first in (True, False):
        f, disagreement = _focal_from_homography(centered, long_first)
        options.append((long_first, f, disagreement))
    plausible = [o for o in options if o[1] is not None and 0.35 * w < o[1] < 3.0 * w and o[2] < 0.25]
    if focal_px_override:
        f_used, focal_source = float(focal_px_override), "cli"
        aspect = _aspect_with_focal(centered, f_used)
        long_first = aspect > 1.0
    elif plausible:
        long_first, f_used, disagreement = min(plausible, key=lambda o: o[2])
        focal_source = "a4_homography"
        info["focal_constraint_disagreement_pct"] = round(100 * (np.exp(disagreement) - 1), 2)
    else:
        f_used, focal_source = f_prior, "hfov_prior"
        aspect = _aspect_with_focal(centered, f_used)
        long_first = aspect > 1.0
        info["focal_fallback_reason"] = "A4 too fronto-parallel or inconsistent for a closed-form focal"
    info["focal_candidates_px"] = [None if o[1] is None else round(o[1], 1) for o in options]

    model2d = _model_points(long_first)
    model3d = np.column_stack([model2d, np.zeros(4)])

    def residuals(params: np.ndarray) -> np.ndarray:
        f, rvec, tvec = params[0], params[1:4], params[4:7]
        K = np.array([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.0]])
        proj, _ = cv2.projectPoints(model3d, rvec, tvec, K, None)
        res = (proj.reshape(-1, 2) - corners_px).ravel()
        if focal_source != "a4_homography":   # keep a prior/CLI focal from drifting
            res = np.append(res, 50.0 * (f - f_used) / f_used)
        return res

    K0 = np.array([[f_used, 0, pp[0]], [0, f_used, pp[1]], [0, 0, 1.0]])
    ok, rvec, tvec = cv2.solvePnP(model3d, corners_px.astype(np.float64), K0, None, flags=cv2.SOLVEPNP_IPPE)
    sol = least_squares(residuals, np.concatenate([[f_used], rvec.ravel(), tvec.ravel()]), method="lm")
    f_final, rvec, tvec = sol.x[0], sol.x[1:4], sol.x[4:7]
    K = np.array([[f_final, 0, pp[0]], [0, f_final, pp[1]], [0, 0, 1.0]])
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.copy()
    proj, _ = cv2.projectPoints(model3d, rvec, tvec, K, None)
    reproj_rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners_px) ** 2, -1))))

    if not long_first:  # make +x run along the long edge
        R = R @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    calib = Calibration(K=K, R_ct=R, t_ct=t, method="a4", focal_source=focal_source, info=info)
    if calib.camera_center_table()[2] < 0:   # +z must point toward the camera
        calib.R_ct = calib.R_ct @ np.diag([1.0, -1.0, -1.0])
    o, x_end = calib.project_table(np.array([[0, 0, 0], [0.1, 0, 0]]))
    if x_end[0] < o[0]:                       # +x should point right in the image
        calib.R_ct = calib.R_ct @ np.diag([-1.0, -1.0, 1.0])

    cam_center = calib.camera_center_table()
    info.update({
        "a4_corners_px": np.round(corners_px, 2).tolist(),
        "focal_px": round(float(f_final), 2),
        "hfov_deg": round(float(np.rad2deg(2 * np.arctan(w / 2 / f_final))), 2),
        "a4_reprojection_rms_px": round(reproj_rms, 4),
        "a4_aspect_ratio_implied": round(_aspect_with_focal(centered, f_final) ** (1 if long_first else -1), 4),
        "a4_aspect_ratio_true": round(A4_LONG_M / A4_SHORT_M, 4),
        "camera_center_table_m": np.round(cam_center, 4).tolist(),
        "camera_height_above_table_m": round(float(cam_center[2]), 4),
        "camera_distance_to_a4_center_m": round(float(np.linalg.norm(cam_center)), 4),
        "view_elevation_deg": round(float(np.rad2deg(np.arcsin(cam_center[2] / np.linalg.norm(cam_center)))), 2),
    })
    return calib


def calibration_from_plane(K: np.ndarray, normal_cam: np.ndarray, plane_offset: float,
                           origin_hint_cam: np.ndarray, focal_source: str, info: dict) -> Calibration:
    """No-A4 fallback: table frame from a plane n.X = -d (camera frame, n up)."""
    n = normal_cam / np.linalg.norm(normal_cam)
    # foot of the hint point on the plane
    origin = origin_hint_cam - (n @ origin_hint_cam + plane_offset) * n
    x_axis = np.array([1.0, 0, 0]) - n[0] * n
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(n, x_axis)
    R = np.column_stack([x_axis, y_axis, n])
    calib = Calibration(K=K, R_ct=R, t_ct=origin, method="no_a4_fallback", focal_source=focal_source, info=info)
    if calib.camera_center_table()[2] < 0:
        calib.R_ct = calib.R_ct @ np.diag([1.0, -1.0, -1.0])
    cam_center = calib.camera_center_table()
    info.update({"camera_center_table_m": np.round(cam_center, 4).tolist(),
                 "camera_height_above_table_m": round(float(cam_center[2]), 4),
                 "focal_px": round(float(K[0, 0]), 2)})
    return calib
