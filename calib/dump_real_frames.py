"""Dump cam_head / cam_right frames + joint states from the friend's dataset for calibration.
  python calib/dump_real_frames.py --out /work/outputs_large/calib/real --n 24 --seed 0
Picks table-to-bowl episodes (instruction mentions 'table' and 'bowl'), one random frame each,
spread over the episode (start, reach, grasp, carry). Writes <out>/frames.npz (images uint8,
state float32 [n,6], episode, frame_index, task) and PNGs for eyeballing."""
import argparse, json
from pathlib import Path
import numpy as np


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--repo", default="MinhDuk/final_dataset_annotated")
  ap.add_argument("--root", default=None, help="local snapshot dir (else HF cache/download)")
  ap.add_argument("--out", required=True)
  ap.add_argument("--n", type=int, default=24)
  ap.add_argument("--seed", type=int, default=0)
  a = ap.parse_args()
  from lerobot.datasets.lerobot_dataset import LeRobotDataset
  from PIL import Image
  ds = LeRobotDataset(a.repo, root=a.root, video_backend="pyav")
  tasks = ds.meta.tasks  # DataFrame indexed by task string
  task_of = {int(r.task_index): t for t, r in tasks.iterrows()}
  eps = ds.meta.episodes
  rng = np.random.default_rng(a.seed)
  cand = []
  for e in range(ds.meta.total_episodes):
    ep = eps[e]
    t = " ".join(ep["tasks"]) if isinstance(ep["tasks"], list) else str(ep["tasks"])
    if "table" in t.lower() and "bowl" in t.lower():
      cand.append(e)
  pick = rng.choice(cand, size=a.n, replace=False)
  out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
  head, right, st, act, epi, fidx, tk = [], [], [], [], [], [], []
  for e in pick:
    ep = eps[int(e)]
    lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    i = int(rng.integers(lo, hi))
    it = ds[i]
    img = lambda k: (it[k].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    head.append(img("observation.images.cam_head")); right.append(img("observation.images.cam_right"))
    st.append(it["observation.state"].numpy()); act.append(it["action"].numpy())
    epi.append(int(e)); fidx.append(int(it["frame_index"])); tk.append(it["task"])
    Image.fromarray(head[-1]).save(out / f"head_{len(epi)-1:02d}_ep{int(e)}.png")
  np.savez_compressed(out / "frames.npz", head=np.stack(head), right=np.stack(right),
                      state=np.stack(st), action=np.stack(act), episode=np.array(epi),
                      frame_index=np.array(fidx), task=np.array(tk))
  print(json.dumps({"n": len(epi), "candidates": len(cand), "episodes": epi,
                    "state_min": np.min(st, 0).round(1).tolist(), "state_max": np.max(st, 0).round(1).tolist()}))


if __name__ == "__main__":
  main()
