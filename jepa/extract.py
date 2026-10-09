"""Run the frozen encoders over the clips once and cache the features to disk.

  python jepa/extract.py sim    # judge-set episodes + extra sim positives (for realism)
  python jepa/extract.py real   # the friend's real cam_head clips

How frames are picked (documented, fixed before looking at results):
  uniform16   16 frames spread evenly from the first to the last frame of the episode
              (sim: 10 Hz renders of 14-26 s episodes, so ~1-1.7 s apart; real: 30 Hz
              clips of 14-28 s, same rule on the decoded frames).
  last        the episode's last frame copied 16 times (a still "video"): what V-JEPA
              sees when it has no motion at all. This is the last-frame baseline.
Features per clip: (8, 1024) V-JEPA time-slot means (see vjepa.py).
DINOv2-small (a 22M-parameter image model) also encodes the last frame: 384-d CLS token
plus 384-d mean patch token = 768 numbers. That is a second, non-JEPA last-frame baseline.

Outputs in /work/outputs_large/jepa/:
  feats_sim.npz   ids, front_uniform16, front_last, wrist_uniform16, dino_last
  feats_simfd.npz ids, front_uniform16 for extra accepted base episodes (realism score)
  feats_real.npz  episode_index, cam_head_uniform16
  feats_wrist_last.npz  ids, dino_last (wrist camera, CPU; python jepa/extract.py wrist_last)
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jepa import vjepa  # noqa: E402

OUT = Path("/work/outputs_large/jepa")
BASE = Path("/work/outputs_large/datagen/sugar_v10_n1000")
N_SIMFD = 400  # accepted base episodes for the realism score (two halves of 200)
N_REAL = 400   # real episodes (two halves of 200)
REAL_ROOT = "/work/hf_home/hub/datasets--MinhDuk--final_dataset_annotated/snapshots/*"


def sim_clips(path: str) -> tuple[np.ndarray, np.ndarray]:
  v = vjepa.read_video(path)
  return v[vjepa.uniform_indices(len(v))], np.repeat(v[-1:], vjepa.N_FRAMES, 0)


def load_dino(device="cuda"):
  from transformers import AutoModel
  snap = glob.glob("/work/hf_home/hub/models--facebook--dinov2-small/snapshots/*")[0]
  return AutoModel.from_pretrained(snap, dtype=torch.bfloat16).to(device).eval()


@torch.no_grad()
def dino_encode(model, frames: list[np.ndarray], device="cuda", dtype=torch.bfloat16) -> np.ndarray:
  """Last frames (H, W, 3) uint8 -> (N, 768): CLS + mean patch token at 224x224."""
  out = []
  for i in range(0, len(frames), 32):
    x = torch.stack([vjepa.preprocess(f[None])[0] for f in frames[i:i + 32]])
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", antialias=True)
    h = model(pixel_values=x.to(device, dtype)).last_hidden_state.float()
    out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu().numpy())
  return np.concatenate(out, 0)


def run_sim() -> None:
  eps = json.loads((OUT / "judge_set.json").read_text())
  model = vjepa.load_model()
  dino = load_dino()
  uni, last, wri, lastf = [], [], [], []
  store = {"u": [], "l": [], "w": [], "d": []}
  t0 = time.time()
  for k, e in enumerate(eps):
    u, l = sim_clips(e["front"])
    uni.append(u)
    last.append(l)
    lastf.append(l[0])
    w = vjepa.read_video(e["wrist"])
    wri.append(w[vjepa.uniform_indices(len(w))])
    if len(uni) == 32 or k == len(eps) - 1:
      for name, c in (("u", uni), ("l", last), ("w", wri)):
        store[name].append(vjepa.encode(model, c))
      store["d"].append(dino_encode(dino, lastf))
      uni, last, wri, lastf = [], [], [], []
      print(f"sim {k + 1}/{len(eps)} {time.time() - t0:.0f}s", flush=True)
  s = store
  np.savez(OUT / "feats_sim.npz", ids=np.array([e["id"] for e in eps]),
           front_uniform16=np.concatenate(s["u"]), front_last=np.concatenate(s["l"]),
           wrist_uniform16=np.concatenate(s["w"]), dino_last=np.concatenate(s["d"]))
  # Extra accepted base episodes for the realism score (front, uniform16 only).
  rows = [json.loads(l) for l in (BASE / "attempts.jsonl").read_text().splitlines() if l.strip()]
  acc = sorted(r["attempt"] for r in rows if r["accepted"])[:N_SIMFD]
  feats = []
  for i in range(0, len(acc), 32):
    clips = [sim_clips(str(BASE / "staging" / f"attempt_{a:06d}.front.mp4"))[0] for a in acc[i:i + 32]]
    feats.append(vjepa.encode(model, clips))
  np.savez(OUT / "feats_simfd.npz", ids=np.array(acc), front_uniform16=np.concatenate(feats))
  print(f"sim done: {len(eps)} judge episodes, {len(acc)} realism episodes, {time.time() - t0:.0f}s")


def real_episode_clip(args) -> np.ndarray:
  """Decode one real episode's cam_head span and keep 16 uniformly spaced frames."""
  path, t_from, t_to = args
  import av
  frames = []
  with av.open(path) as c:
    st = c.streams.video[0]
    st.thread_type = "AUTO"
    c.seek(int(max(t_from - 1.0, 0) / st.time_base), stream=st, backward=True)
    for f in c.decode(st):
      t = float(f.pts * st.time_base)
      if t < t_from - 1e-3:
        continue
      if t >= t_to - 1e-3:
        break
      frames.append(f.to_ndarray(format="rgb24"))
  v = np.stack(frames)
  return v[vjepa.uniform_indices(len(v))]


def run_real() -> None:
  import pandas as pd
  root = glob.glob(REAL_ROOT)[0]
  e = pd.read_parquet(f"{root}/meta/episodes/chunk-000/file-000.parquet")
  pick = np.sort(np.random.default_rng(0).choice(len(e), N_REAL, replace=False))
  key = "videos/observation.images.cam_head"
  jobs = []
  for i in pick:
    r = e.iloc[i]
    path = f"{root}/videos/observation.images.cam_head/chunk-{int(r[key + '/chunk_index']):03d}/" \
           f"file-{int(r[key + '/file_index']):03d}.mp4"
    jobs.append((path, float(r[key + "/from_timestamp"]), float(r[key + "/to_timestamp"])))
  t0 = time.time()
  workers = int(os.environ.get("JEPA_WORKERS", "4"))
  with ProcessPoolExecutor(workers) as ex:
    clips = list(ex.map(real_episode_clip, jobs, chunksize=4))
  print(f"decoded {len(clips)} real clips in {time.time() - t0:.0f}s; shape {clips[0].shape}")
  model = vjepa.load_model()
  feats = vjepa.encode(model, clips)
  np.savez(OUT / "feats_real.npz", episode_index=e["episode_index"].values[pick],
           cam_head_uniform16=feats)
  # Save one contact sheet's worth of frames for a visual sanity check.
  np.save(OUT / "real_example_clip.npy", clips[0])
  print(f"real done in {time.time() - t0:.0f}s")


def run_wrist_last() -> None:
  """CPU only (added after the GPU budget was spent): DINOv2-small on the WRIST camera's
  last frame, the fair last-frame baseline for the wrist-video variant."""
  torch.set_num_threads(int(os.environ.get("JEPA_WORKERS", "6")))
  eps = json.loads((OUT / "judge_set.json").read_text())
  dino = load_dino("cpu").float()
  frames = [vjepa.read_video(e["wrist"])[-1] for e in eps]
  with torch.autocast("cpu", enabled=False):
    feats = dino_encode(dino, frames, device="cpu", dtype=torch.float32)
  np.savez(OUT / "feats_wrist_last.npz", ids=np.array([e["id"] for e in eps]), dino_last=feats)
  print(f"wrist last-frame DINO: {feats.shape}")


if __name__ == "__main__":
  {"sim": run_sim, "real": run_real, "wrist_last": run_wrist_last}[sys.argv[1]]()
