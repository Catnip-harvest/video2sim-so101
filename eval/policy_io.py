"""Load a LeRobot `pretrained_model/` checkpoint and run it one observation at a time.

The observation is built exactly the way LeRobot's own robot-inference path builds it
(`lerobot.policies.utils.prepare_observation_for_inference`): uint8 HxWx3 images become
float32 [0, 1] CxHxW tensors with a batch dim, which is also what LeRobotDataset hands the
policy at training time (decoded video frames / 255). The state is float32 [6] in the
MolmoAct2 joint frame (degrees, old LeRobot calibration), like the training dataset.
The environment state (cube pose) is deliberately NOT given to the policy.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

TASK = "pick up the cube and place it in the bin"
CAMERA_KEYS = ("observation.images.cam_head", "observation.images.cam_right")


def load_policy(path: str | Path, device: str = "cuda", inference_action_mode: str = "continuous"):
  """Return (policy, preprocessor, postprocessor) for a lerobot pretrained_model dir."""
  import torch  # noqa: F401
  from lerobot.configs.policies import PreTrainedConfig
  from lerobot.policies.factory import get_policy_class, make_pre_post_processors

  path = str(Path(path).resolve())
  cfg = PreTrainedConfig.from_pretrained(path)
  cfg.pretrained_path = path
  cfg.device = device
  if hasattr(cfg, "inference_action_mode"):
    cfg.inference_action_mode = inference_action_mode
  if hasattr(cfg, "enable_inference_cuda_graph"):
    cfg.enable_inference_cuda_graph = False  # CUDA graphs fail on ROCm
  policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
  policy.to(device)
  policy.eval()
  pre, post = make_pre_post_processors(
    policy_cfg=cfg, pretrained_path=path,
    preprocessor_overrides={"device_processor": {"device": device}})
  return policy, pre, post


class PolicyRunner:
  """select_action wrapper: one 10 Hz observation in, one 6-D action (MolmoAct2 frame) out."""

  def __init__(self, path: str | Path, device: str = "cuda", task: str = TASK):
    import torch
    self.torch = torch
    self.device = device
    self.task = task
    self.policy, self.pre, self.post = load_policy(path, device)
    self.chunk_ms: list[float] = []  # wall time of each call that generated a new chunk
    self.call_ms: list[float] = []

  def reset(self) -> None:
    self.policy.reset()
    # Processor steps can carry state between calls (relative/absolute action steps).
    for p in (self.pre, self.post):
      if hasattr(p, "reset"):
        p.reset()

  def needs_obs(self) -> bool:
    """True when the next call generates a new chunk (only then is the observation read)."""
    return len(getattr(self.policy, "_action_queue", ())) == 0

  def act(self, state_mm2: np.ndarray, images: dict[str, np.ndarray] | None) -> np.ndarray:
    from lerobot.policies.utils import prepare_observation_for_inference
    torch = self.torch
    obs = {"observation.state": np.asarray(state_mm2, np.float32)}
    if self.needs_obs():
      for k in CAMERA_KEYS:
        img = images[k]
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, (k, img.shape, img.dtype)
        obs[k] = img
    new_chunk = self.needs_obs()
    t0 = time.perf_counter()
    if new_chunk:
      batch = prepare_observation_for_inference(obs, torch.device(self.device), self.task)
      batch = self.pre(batch)
      self._last_batch = batch
    else:
      # select_action only pops the queued chunk and ignores the batch; skip the ~180 ms
      # of tokenisation/image packing the preprocessor would spend on it.
      batch = self._last_batch
    with torch.inference_mode():
      a = self.policy.select_action(batch)
    a = self.post(a)
    if torch.is_tensor(a):
      a = a.detach().float().cpu().numpy()
    a = np.asarray(a, np.float64).reshape(-1)[:6]
    ms = (time.perf_counter() - t0) * 1e3
    self.call_ms.append(ms)
    if new_chunk:
      self.chunk_ms.append(ms)
    return a
