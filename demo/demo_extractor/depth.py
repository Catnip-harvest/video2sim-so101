"""Monocular depth with Depth Anything V2 Small (Apache-2.0; the Base/Large
checkpoints are CC-BY-NC and are not used).

The model outputs relative inverse depth ("disparity") known only up to a scale
and a shift. Where the table plane is known (A4 calibration), each frame is made
metric by fitting   1/Z_true = a * disparity + b   on table pixels whose true
depth follows from the calibration. The result is used as an independent check
on the object's depth while it is carried, and (without an A4 sheet) to find
the table plane at all.
"""
from __future__ import annotations

import gc
from pathlib import Path

import cv2
import numpy as np


class DepthModel:
    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()

    def disparity(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self.torch.inference_mode():
            inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
            pred = self.model(**inputs).predicted_depth  # (1, h, w), relative inverse depth
            pred = self.torch.nn.functional.interpolate(pred[:, None], size=rgb.shape[:2], mode="bilinear",
                                                        align_corners=False)[0, 0]
        return pred.cpu().numpy().astype(np.float64)

    def close(self) -> None:
        del self.model
        gc.collect()


def table_depth_map(calib, shape: tuple[int, int]) -> np.ndarray:
    """True camera depth Z of the table plane (z=0) at every pixel; NaN if behind."""
    h, w = shape
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(uu - calib.K[0, 2]) / calib.K[0, 0], (vv - calib.K[1, 2]) / calib.K[1, 1], np.ones_like(uu)], -1)
    normal_cam = calib.R_ct[:, 2]                 # table z axis in camera frame
    point_cam = calib.t_ct                        # table origin in camera frame
    denom = rays @ normal_cam
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (normal_cam @ point_cam) / denom      # depth along the optical axis (ray has z=1)
    z[(z <= 0) | ~np.isfinite(z)] = np.nan
    return z


def align_to_table(disparity: np.ndarray, table_z: np.ndarray, table_mask: np.ndarray) -> tuple[float, float, float]:
    """Robust fit 1/Z = a*disp + b on table pixels. Returns (a, b, relative rms)."""
    sel = table_mask & np.isfinite(table_z)
    if sel.sum() < 200:
        return np.nan, np.nan, np.nan
    d = disparity[sel]
    inv = 1.0 / table_z[sel]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d), size=min(len(d), 20000), replace=False)
    d, inv = d[idx], inv[idx]
    best = None
    for _ in range(200):
        i, j = rng.choice(len(d), 2, replace=False)
        if abs(d[i] - d[j]) < 1e-6:
            continue
        a = (inv[i] - inv[j]) / (d[i] - d[j])
        b = inv[i] - a * d[i]
        err = np.abs(a * d + b - inv) / inv
        inliers = err < 0.03
        if best is None or inliers.sum() > best[0].sum():
            best = (inliers, a, b)
    if best is None:
        return np.nan, np.nan, np.nan
    inliers = best[0]
    A = np.column_stack([d[inliers], np.ones(inliers.sum())])
    (a, b), *_ = np.linalg.lstsq(A, inv[inliers], rcond=None)
    rel_rms = float(np.sqrt(np.mean(((a * d[inliers] + b) - inv[inliers]) ** 2 / inv[inliers] ** 2)))
    return float(a), float(b), rel_rms
