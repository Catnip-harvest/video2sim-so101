"""Control A for filtered BC: the same NUMBER of fresh scripted-expert episodes on the SAME
scenes as the kept policy rollouts, exported exactly like the SFT dataset.

  python eval/control_a.py --kept DATASET_DIR/meta/kept_ids.json --out RUN_DIR \
      --dataset-dir CONTROL_A_DATASET [--workers 6]

It is generate.py with the rollouts' --seed (2000) and the rollouts' attempt ids: the same
worker (generate._attempt: sample_variation(seed, id, variety=True) -> expert -> expert
verifier -> render accepted, quality fast, 256x256, 10 Hz) and the same finalize (MolmoAct2
frame, cam_head/cam_right). Then datagen/use_pretrained_norm.py.
Same scenes first: every kept id is attempted. If the expert fails some of them, the count
is topped up with further ids of the same seed (not-kept rollout ids first, ascending, then
ids after the rollout range), so N(control A) = N(kept rollouts). The report says how many.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.setdefault("LP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
DATAGEN = ROOT / "datagen"
sys.path.insert(0, str(DATAGEN))

import generate as G  # noqa: E402

SPEC = ROOT / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--kept", required=True, help="kept_ids.json from rollouts_to_dataset.py")
  ap.add_argument("--rollout-range", type=int, default=None,
                  help="number of rollout attempt ids (0..R-1) run for data; top-ups prefer the "
                       "not-kept ones among them (default: max kept id + 1)")
  ap.add_argument("--out", required=True, help="generate.py-style run dir (staging, logs)")
  ap.add_argument("--dataset-dir", required=True)
  ap.add_argument("--workers", type=int, default=6)
  ap.add_argument("--no-pretrained-norm", action="store_true")
  a = ap.parse_args()
  kept = json.loads(Path(a.kept).read_text())
  seeds = {s for s, _ in kept["ids"]}
  if len(seeds) != 1:
    sys.exit(f"kept ids span several base seeds {seeds}; run control A per seed")
  seed = seeds.pop()
  ids = sorted(i for _, i in kept["ids"])
  target = len(ids)
  rng = a.rollout_range or (max(ids) + 1)
  topups = [i for i in range(rng) if i not in set(ids)] + list(range(rng, rng + 4 * target + 50))
  out = Path(a.out).resolve()
  out.mkdir(parents=True, exist_ok=True)
  cfg = {
    "spec": str(SPEC), "n": target, "seed": seed, "render": True, "size": [256, 256],
    "codec": "h264", "task": G.DEFAULT_TASK, "out": str(out),
    "video_keys": ["observation.images.front", "observation.images.wrist"],
    "pinch_width": G.pinch_width_from_spec(SPEC), "no_dataset": False, "servo_hold": "foh",
    "render_quality": "fast", "variety": True, "record_hz": 10, "keep_rejects": False,
  }
  (out / "run.json").write_text(json.dumps(cfg, indent=2))
  cfg["export"] = {"frame": "molmoact2", "arm_calib": None, "dataset_dir": str(Path(a.dataset_dir).resolve()),
                   "camera_names": {"front": "cam_head", "wrist": "cam_right"}}
  rows = G.load_log(out)  # resumable
  log = open(out / "attempts.jsonl", "a", encoding="utf-8")
  t0 = time.perf_counter()

  def run(batch):
    todo = [i for i in batch if i not in rows]
    if not todo:
      return
    with mp.get_context("spawn").Pool(a.workers, initializer=G._init_worker, initargs=(cfg,)) as pool:
      for r in pool.imap_unordered(G._attempt, todo):
        rows[r["attempt"]] = r
        log.write(json.dumps({k: r[k] for k in r if k != "image_stats"}) + "\n")
        log.flush()

  run(ids)
  chosen = [i for i in ids if rows[i]["accepted"]]
  same_scene = len(chosen)
  pos = 0
  while len(chosen) < target and pos < len(topups):
    need = target - len(chosen)
    batch = topups[pos:pos + max(need, a.workers)]
    pos += len(batch)
    run(batch)
    for i in batch:
      if rows[i]["accepted"] and len(chosen) < target:
        chosen.append(i)
  log.close()
  if len(chosen) < target:
    sys.exit(f"only {len(chosen)} of {target} expert episodes accepted")
  sel = {i: rows[i] for i in chosen}
  G.finalize(out, cfg, sel, time.perf_counter() - t0, a.workers)
  ds = Path(a.dataset_dir)
  if not a.no_pretrained_norm:
    subprocess.run([sys.executable, str(DATAGEN / "use_pretrained_norm.py"), str(ds)], check=True,
                   stdout=subprocess.DEVNULL)
  rep = {"seed": seed, "target": target, "same_scene_accepted": same_scene,
         "same_scene_failed": [i for i in ids if not rows[i]["accepted"]],
         "topup_ids": [i for i in chosen if i not in set(ids)], "dataset": str(ds),
         "wall_s": round(time.perf_counter() - t0, 1), "workers": a.workers}
  (ds / "meta" / "control_a.json").write_text(json.dumps(rep, indent=1))
  print(json.dumps(rep, indent=1))


if __name__ == "__main__":
  main()
