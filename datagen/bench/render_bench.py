"""Measure OSMesa render throughput on staged episodes.
  python bench/render_bench.py --run runs/sugar_n20 --workers 19 --frames 150 --size 256x256 --every 1
Each worker renders the first --frames frames of one staged episode (front + wrist,
encoding included) and the script reports per-worker and total frames/s."""
import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def job(a):
  i, run, frames, size, every, tmp, qual = a
  import numpy as np
  from r2s.paths import load_task_spec
  from r2s.variations import sample_variation
  from r2s.render import render_episode
  cfg = json.loads((Path(run) / "run.json").read_text())
  spec = load_task_spec(cfg["spec"])
  stems = sorted((Path(run) / "staging").glob("attempt_*.npz"))
  stem = stems[i % len(stems)]
  aid = int(stem.stem.split("_")[1])
  v = sample_variation(spec, cfg["seed"], aid)
  q = np.load(stem)["qpos_full"][:frames][::every]
  keys = ["observation.images.front", "observation.images.wrist"]
  outs = {k: f"{tmp}/w{i}.{k.split('.')[-1]}.mp4" for k in keys}
  t0 = time.perf_counter()
  render_episode(v, q, keys, outs, size=size, codec="h264", object_kind=spec.object.kind, quality=qual)
  dt = time.perf_counter() - t0
  return len(q) * len(keys), dt


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", required=True)
  ap.add_argument("--workers", type=int, default=1)
  ap.add_argument("--frames", type=int, default=150)
  ap.add_argument("--size", default="256x256")
  ap.add_argument("--every", type=int, default=1)
  ap.add_argument("--quality", default="full")
  a = ap.parse_args()
  size = tuple(int(x) for x in a.size.split("x"))
  tmp = "/tmp/render_bench"; os.makedirs(tmp, exist_ok=True)
  t0 = time.perf_counter()
  with mp.get_context("spawn").Pool(a.workers) as p:
    res = p.map(job, [(i, a.run, a.frames, size, a.every, tmp, a.quality) for i in range(a.workers)])
  wall = time.perf_counter() - t0
  imgs = sum(r[0] for r in res)
  per = [r[0] / r[1] for r in res]
  print(json.dumps({"workers": a.workers, "size": a.size, "every": a.every, "quality": a.quality,
                    "LP_NUM_THREADS": os.environ.get("LP_NUM_THREADS"),
                    "GALLIUM_DRIVER": os.environ.get("GALLIUM_DRIVER"),
                    "images": imgs, "wall_s": round(wall, 1),
                    "images_per_s_total": round(imgs / wall, 1),
                    "images_per_s_per_worker_mean": round(sum(per) / len(per), 1)}))
