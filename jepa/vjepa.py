"""Frozen V-JEPA 2 ViT-L video encoder: frames in, one feature block per clip out.

Model: facebook/vjepa2-vitl-fpc64-256 (MIT), loaded with transformers' AutoModel and used
through ``model.get_vision_features`` (the encoder only; the predictor is never used).
Nothing is trained here: the weights stay frozen, we run it in bf16 under no_grad.

Input clip: 16 RGB frames. Preprocessing is the model's own eval recipe
(video_preprocessor_config.json): resize the short edge to 292 px (bilinear), centre-crop
256x256, scale to [0, 1], normalise with the ImageNet mean/std. The same function is used
for sim renders (256x256) and real camera frames (640x480), so both see the same pipeline.

Output tokens: the encoder cuts the clip into 2-frame x 16x16-pixel "tubelets", so 16
frames give 8 time slots x 16x16 = 256 spatial patches = 2048 tokens of 1024 numbers.
We keep, per clip, the mean over the 256 spatial tokens of each time slot: an (8, 1024)
array. "Mean-pooled" features = the mean of those 8 rows (1024 numbers per clip).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

REPO = "facebook/vjepa2-vitl-fpc64-256"
N_FRAMES = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(device: str = "cuda"):
  """Frozen encoder in bf16 (about 0.6 GB of weights)."""
  from transformers import AutoModel
  snap = glob.glob(os.path.join(os.environ.get("HF_HOME", "/work/hf_home"),
                                "hub/models--facebook--vjepa2-vitl-fpc64-256/snapshots/*"))
  model = AutoModel.from_pretrained(snap[0] if snap else REPO, torch_dtype=torch.bfloat16,
                                    attn_implementation="sdpa")
  return model.to(device).eval()


def uniform_indices(n: int, k: int = N_FRAMES) -> np.ndarray:
  """k frame indices spread evenly from the first to the last frame (inclusive)."""
  return np.linspace(0, n - 1, k).round().astype(int)


def preprocess(frames: np.ndarray) -> torch.Tensor:
  """(T, H, W, 3) uint8 -> (T, 3, 256, 256) float, the model's eval preprocessing."""
  x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
  h, w = x.shape[-2:]
  s = 292 / min(h, w)
  x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", antialias=True,
                    align_corners=False)
  h, w = x.shape[-2:]
  top, left = (h - 256) // 2, (w - 256) // 2
  x = x[..., top:top + 256, left:left + 256]
  return (x - MEAN) / STD


@torch.no_grad()
def encode(model, clips: list[np.ndarray], device: str = "cuda", batch: int = 8) -> np.ndarray:
  """clips: list of (16, H, W, 3) uint8 arrays -> (N, 8, 1024) float16 time-slot features."""
  out = []
  for i in range(0, len(clips), batch):
    x = torch.stack([preprocess(c) for c in clips[i:i + batch]]).to(device, torch.bfloat16)
    tok = model.get_vision_features(x)  # (B, 8*256, 1024)
    b, n, d = tok.shape
    t = x.shape[1] // model.config.tubelet_size
    out.append(tok.float().view(b, t, n // t, d).mean(2).cpu().numpy().astype(np.float16))
  return np.concatenate(out, 0)


def read_video(path: str) -> np.ndarray:
  """All frames of a (short) mp4 as (T, H, W, 3) uint8."""
  import av
  with av.open(path) as c:
    return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
