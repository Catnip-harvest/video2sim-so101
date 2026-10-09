"""Object masks through the clip with SAM 2.1 hiera-tiny (Apache-2.0), run via
Hugging Face transformers in streaming mode (one frame in memory at a time).

Prompting:
  * point  - a click on the object in frame 0 (--object-point x,y)
  * box    - a box around the object in frame 0 (--object-box x0,y0,x1,y1)
  * auto   - "the object nearest the hand at grasp time": at the first thumb-index
             closure SAM 2 is asked what is between thumb and index (the other hand
             keypoints are negative clicks); that object is tracked backwards to
             frame 0 (it is static there while the hand moves away), and the
             frame-0 mask seeds the forward pass.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ObjectTrack:
    masks: list[np.ndarray | None]   # bool (H, W) per frame at processing resolution
    scores: np.ndarray               # SAM 2 object-presence logit per frame
    prompt: dict                     # how frame 0 was prompted

    def centroid(self, k: int) -> np.ndarray | None:
        m = self.masks[k]
        if m is None or m.sum() < 5:
            return None
        ys, xs = np.nonzero(m)
        return np.array([xs.mean(), ys.mean()])

    def bbox(self, k: int) -> np.ndarray | None:
        m = self.masks[k]
        if m is None or m.sum() < 5:
            return None
        ys, xs = np.nonzero(m)
        return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)

    def area(self, k: int) -> int:
        m = self.masks[k]
        return 0 if m is None else int(m.sum())


class Sam2Tracker:
    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.torch = torch
        self.device = device
        self.model = Sam2VideoModel.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
        self.processor = Sam2VideoProcessor.from_pretrained(str(model_dir), local_files_only=True)

    def close(self) -> None:
        del self.model
        gc.collect()

    def _new_session(self):
        return self.processor.init_video_session(inference_device=self.device, dtype=self.torch.float32)

    def _to_mask(self, output, original_size) -> tuple[np.ndarray, float]:
        masks = self.processor.post_process_masks([output.pred_masks], original_sizes=[original_size],
                                                  binarize=False)[0]
        logit = float(output.object_score_logits.reshape(-1)[0])
        return (masks[0, 0] > 0.0).cpu().numpy(), logit

    def segment_single(self, frame_rgb: np.ndarray, points=None, labels=None, box=None) -> np.ndarray:
        """One-frame segmentation with point and/or box prompts (pixels)."""
        with self.torch.inference_mode():
            session = self._new_session()
            inputs = self.processor(images=frame_rgb, device=self.device, return_tensors="pt")
            original_size = tuple(int(v) for v in inputs.original_sizes[0])
            kwargs = {}
            if points is not None:
                kwargs["input_points"] = [[[list(map(float, p)) for p in points]]]
                kwargs["input_labels"] = [[list(map(int, labels))]]
            if box is not None:
                kwargs["input_boxes"] = [[list(map(float, box))]]
            self.processor.add_inputs_to_inference_session(
                inference_session=session, frame_idx=0, obj_ids=1, original_size=original_size, **kwargs)
            output = self.model(inference_session=session, frame=inputs.pixel_values[0])
            mask, _ = self._to_mask(output, original_size)
        return mask

    def track(self, frames_rgb: list[np.ndarray], points=None, labels=None, box=None, mask=None,
              progress=None) -> tuple[list[np.ndarray], np.ndarray]:
        masks, logits = [], []
        with self.torch.inference_mode():
            session = self._new_session()
            for k, frame in enumerate(frames_rgb):
                inputs = self.processor(images=frame, device=self.device, return_tensors="pt")
                original_size = tuple(int(v) for v in inputs.original_sizes[0])
                if k == 0 and mask is not None:
                    self.processor.add_inputs_to_inference_session(
                        inference_session=session, frame_idx=0, obj_ids=1, input_masks=mask.astype(bool))
                elif k == 0:
                    kwargs = {}
                    if points is not None:
                        kwargs["input_points"] = [[[list(map(float, p)) for p in points]]]
                        kwargs["input_labels"] = [[list(map(int, labels))]]
                    if box is not None:
                        kwargs["input_boxes"] = [[list(map(float, box))]]
                    self.processor.add_inputs_to_inference_session(
                        inference_session=session, frame_idx=0, obj_ids=1, original_size=original_size, **kwargs)
                # explicit frame_idx: pruning old frames must not make indices collide
                output = self.model(inference_session=session, frame_idx=k, frame=inputs.pixel_values[0])
                mask, logit = self._to_mask(output, original_size)
                masks.append(mask)
                logits.append(logit)
                self._prune_session(session, k)
                if progress and k % 10 == 0:
                    progress(f"sam2 frame {k}/{len(frames_rgb)}")
        return masks, np.array(logits)

    @staticmethod
    def _prune_session(session, k: int, keep: int = 24) -> None:
        """Streaming mode keeps every processed frame; drop what memory attention no longer reads."""
        frames = getattr(session, "processed_frames", None)
        if frames:
            for old in [i for i in frames if 0 < i < k]:
                del frames[old]
        for per_obj in getattr(session, "output_dict_per_obj", {}).values():
            non_cond = per_obj.get("non_cond_frame_outputs", {})
            for old in [i for i in non_cond if i < k - keep]:
                del non_cond[old]


def build_prompt(mode: str, frames: list[np.ndarray], tracker: Sam2Tracker, point=None, box=None,
                 grasp_frame: int | None = None, grasp_pinch_px=None, hand_px_at_grasp=None) -> dict:
    """Return {'points', 'labels', 'box', 'mode', ...} for frame 0."""
    import cv2

    if mode == "point":
        return {"mode": "point", "points": [point], "labels": [1], "box": None}
    if mode == "box":
        return {"mode": "box", "points": None, "labels": None, "box": box}
    # auto: what is between thumb and index at the grasp candidate frame? Segment it
    # there (hand keypoints as negative clicks), then track it BACKWARDS to frame 0
    # (the object is static before the grasp while the hand moves away), and seed
    # the forward pass with that frame-0 mask.
    if grasp_frame is None or grasp_pinch_px is None:
        raise RuntimeError("auto object prompt needs a grasp candidate from the hand track; "
                           "pass --object-point or --object-box")
    keep_positive = {2, 3, 4, 6, 7, 8}          # thumb and index joints near the pinch
    negatives = [] if hand_px_at_grasp is None else         [hand_px_at_grasp[i] for i in range(21) if i not in keep_positive and np.isfinite(hand_px_at_grasp[i, 0])]
    points = [list(map(float, grasp_pinch_px))] + [list(map(float, p)) for p in negatives]
    labels = [1] + [0] * len(negatives)
    stride = max(1, grasp_frame // 12)
    order = list(range(grasp_frame, -1, -stride))
    if order[-1] != 0:
        order.append(0)
    frames_rgb = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in order]
    masks, _ = tracker.track(frames_rgb, points=points, labels=labels)
    grasp_mask, mask0 = masks[0], masks[-1]
    h, w = mask0.shape
    if mask0.sum() < 20 or mask0.sum() > 0.10 * h * w:
        raise RuntimeError(f"auto object prompt failed (frame-0 mask of {int(mask0.sum())} px); "
                           "pass --object-point or --object-box")
    ys, xs = np.nonzero(mask0)
    touches_border = xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2
    overlap = float((mask0 & grasp_mask).sum()) / max(1.0, float(grasp_mask.sum()))
    return {"mode": "auto", "points": None, "labels": None, "box": None, "mask": mask0,
            "grasp_frame_used": int(grasp_frame), "grasp_mask_px": int(grasp_mask.sum()),
            "frame0_mask_px": int(mask0.sum()), "frame0_mask_touches_border": bool(touches_border),
            "grasp_mask_overlap_with_frame0": round(overlap, 3), "backward_frames": len(order)}


def change_blob_prompt(frames: list[np.ndarray], hand_px: np.ndarray, grasp_frame: int,
                       pinch_px: np.ndarray) -> dict | None:
    """With a still camera, the picked object's start spot is where the hand-free
    first frames and hand-free last frames differ. Of the changed blobs, take the
    one nearest the pinch at the grasp; prompt SAM 2 on frame 0 with its box."""
    import cv2

    T = len(frames)
    h, w = frames[0].shape[:2]
    hand_ok = np.isfinite(hand_px[:, 0, 0])
    first = [k for k in range(min(T, grasp_frame)) if not hand_ok[k]][:5] or [0]
    last = [k for k in range(T - 1, grasp_frame, -1) if not hand_ok[k]][:5] or [T - 1]

    def still(ks):
        return np.median(np.stack([cv2.GaussianBlur(frames[k], (5, 5), 0) for k in ks]), axis=0).astype(np.int16)

    a, b = still(first), still(last)
    shift = camera_shift_px(a.astype(np.uint8), b.astype(np.uint8))
    if np.hypot(*shift) > 0.5:   # undo a small camera drift before differencing
        M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
        b = cv2.warpAffine(b.astype(np.uint8), M, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.int16)
    diff = np.abs(a - b).max(-1).astype(np.uint8)
    margin = int(np.ceil(np.hypot(*shift))) + 2
    diff[:margin], diff[-margin:], diff[:, :margin], diff[:, -margin:] = 0, 0, 0, 0
    otsu, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    changed = (diff > max(25, otsu)).astype(np.uint8)
    for k in set(first) | set(last):            # never call the hand itself a change
        if hand_ok[k]:
            hull = cv2.convexHull(np.round(hand_px[k]).astype(np.int32))
            cv2.fillConvexPoly(changed, hull, 0)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(changed)
    best = None
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30 or area > 0.10 * h * w:
            continue
        ys, xs = np.nonzero(labels == i)
        if xs.min() <= margin + 1 or ys.min() <= margin + 1 or xs.max() >= w - margin - 2 or ys.max() >= h - margin - 2:
            continue   # the object starts fully in view; border blobs are parallax or lighting
        dist = float(np.min(np.hypot(xs - pinch_px[0], ys - pinch_px[1])))
        if best is None or dist < best[0]:
            best = (dist, i, xs, ys)
    if best is None or best[0] > 0.25 * np.hypot(w, h):
        return None
    dist, i, xs, ys = best
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    pad_x, pad_y = 0.2 * (x1 - x0 + 1) + 3, 0.2 * (y1 - y0 + 1) + 3
    box = [float(max(0, x0 - pad_x)), float(max(0, y0 - pad_y)), float(min(w - 1, x1 + pad_x)),
           float(min(h - 1, y1 + pad_y))]
    j = int(np.argmax(diff[ys, xs]))
    return {"mode": "auto_change_blob", "points": [[float(xs[j]), float(ys[j])]], "labels": [1], "box": box,
            "grasp_frame_used": int(grasp_frame), "change_blob_px": int(len(xs)),
            "change_blob_distance_to_pinch_px": round(dist, 1), "hand_free_frames": [first, last],
            "camera_shift_first_to_last_px": [round(float(shift[0]), 2), round(float(shift[1]), 2)]}


def camera_shift_px(a_bgr: np.ndarray, b_bgr: np.ndarray) -> np.ndarray:
    """Global image translation from a to b (phase correlation), in pixels."""
    import cv2

    ga = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb, win)
    return np.array([dx, dy])
