"""Verifier-passed rollouts -> LeRobot v3 training dataset, in the same format as the SFT set
(/work/outputs_large/datagen/sugar_v10_n1000_molmoact2): 10 Hz, MolmoAct2 joint frame,
cam_head (= sim front) / cam_right (= sim wrist_cam) 256x256 h264, task string, the same
writer (datagen/r2s/lerobot_io.write_dataset), then the checkpoint's own normalisation
stats (datagen/use_pretrained_norm.py).

  python eval/rollouts_to_dataset.py --runs OUT_ROLLOUT_DIR [...] --out DATASET_DIR [--workers 6]

Input: sim_eval.py output dirs (episodes.jsonl + rollouts/*.npz). Kept: rows with
pass (= L_full) True, unless --select L_task. Per kept episode, per 10 Hz step:
  observation.state              the state the policy observed (MolmoAct2 frame)
  action                         the goal the harness commanded (policy action, clipped to the
                                 joint range), MolmoAct2 frame
  observation.environment_state  object pos + quat at the step start
  images                         RE-RENDERED offline from the recorded full qpos with datagen's
                                 renderer (r2s/render.render_episode, quality fast), exactly as
                                 datagen renders its accepted episodes; policy-time frames are
                                 not reused.
Writes DATASET_DIR/meta/kept_ids.json (seed, attempt ids) for control A.
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
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "datagen", ROOT / "calib", ROOT / "eval"):
  if str(p) not in sys.path:
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

TASK = "pick up the cube and place it in the bin"
SPEC = ROOT / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"
SRC_KEYS = ["observation.images.front", "observation.images.wrist"]
RENAME = {"observation.images.front": "observation.images.cam_head",
          "observation.images.wrist": "observation.images.cam_right"}


def _render(job) -> dict:
  from r2s.paths import load_task_spec
  from r2s.render import render_episode
  from r2s.variations import Variation
  row, stage = job
  z = np.load(row["rollout_file"])
  v = Variation.from_dict(row["variation"])
  spec = load_task_spec(str(SPEC))
  stem = Path(stage) / f"ep_{row['seed']}_{row['attempt']:06d}"
  paths = {k: f"{stem}.{k.split('.')[-1]}.mp4" for k in SRC_KEYS}
  t0 = time.perf_counter()
  render_episode(v, z["qpos_full"], SRC_KEYS, paths, size=(256, 256), codec="h264",
                 object_kind=spec.object.kind, quality="fast", fps=10)
  return {"key": (row["seed"], row["attempt"]), "paths": paths,
          "render_s": round(time.perf_counter() - t0, 2), "frames": int(len(z["qpos_full"]))}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--runs", nargs="+", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--select", default="L_full", choices=["L_full", "L_task"])
  ap.add_argument("--max-episodes", type=int, default=None, help="keep at most this many (lowest ids)")
  ap.add_argument("--workers", type=int, default=6)
  ap.add_argument("--no-pretrained-norm", action="store_true")
  a = ap.parse_args()
  from r2s.lerobot_io import EpisodeData, write_dataset

  rows = []
  for d in a.runs:
    for line in (Path(d) / "episodes.jsonl").read_text().splitlines():
      if line.strip():
        r = json.loads(line)
        r["run_dir"] = str(d)
        rows.append(r)
  key = "pass" if a.select == "L_full" else "L_task"
  kept = sorted([r for r in rows if r.get(key) and r.get("variation") and r.get("rollout_file")],
                key=lambda r: (r["seed"], r["attempt"]))
  seen, uniq = set(), []
  for r in kept:  # one episode per (seed, attempt)
    if (r["seed"], r["attempt"]) not in seen:
      seen.add((r["seed"], r["attempt"]))
      uniq.append(r)
  kept = uniq[: a.max_episodes] if a.max_episodes else uniq
  print(f"[rollouts->dataset] {len(rows)} rollouts, {len(kept)} kept ({a.select})", flush=True)
  if not kept:
    sys.exit("nothing to write")
  out = Path(a.out)
  stage = out.parent / (out.name + "_staging")
  stage.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  with mp.get_context("spawn").Pool(a.workers) as pool:
    rend = {r["key"]: r for r in pool.imap_unordered(_render, [(r, str(stage)) for r in kept])}
  render_wall = time.perf_counter() - t0
  eps = []
  for r in kept:
    z = np.load(r["rollout_file"])
    rr = rend[(r["seed"], r["attempt"])]
    T = len(z["action_mm2"])
    assert len(z["state_mm2"]) == T == len(z["env_state"]) == len(z["qpos_full"]), r["rollout_file"]
    eps.append(EpisodeData(
      state=z["state_mm2"].astype(np.float32), action=z["action_mm2"].astype(np.float32),
      env_state=z["env_state"].astype(np.float32), task=TASK,
      videos={RENAME[k]: Path(p) for k, p in rr["paths"].items()},
      image_stats={},  # like generate.finalize: stats from the decoded videos
      meta={"attempt": r["attempt"], "seed": r["variation"]["seed"], "base_seed": r["seed"],
            "variation": r["variation"], "verification": r["verify_stats"],
            "verdict": {"pass": r["pass"], "L_task": r["L_task"], "reasons": r["reasons"]},
            "source": "policy_rollout" if r["mode"] == "policy" else f"{r['mode']}_rollout",
            "rollout_file": r["rollout_file"], "plan": {}, "phase_starts": {}}))
  cfgs = [json.loads((Path(d) / "config.json").read_text()) for d in a.runs]
  extra = {"generator": "eval/rollouts_to_dataset.py", "runs": [str(d) for d in a.runs],
           "run_configs": cfgs, "select": a.select,
           "units": "MolmoAct2-SO100_101 frame = LeRobot pre-PR #777: lift_old = 90 - lift_new, "
                    "elbow_old = elbow_new + 90; object pose in metres / wxyz quaternion, robot base frame",
           "joint_frame": "molmoact2", "arm_calib": None, "conversion": "calib/so101_units.py",
           "action_semantics": "absolute joint position goal commanded at this 10 Hz step (the "
                               "policy's action, clipped to the joint range); the harness ramps "
                               "the servo goal to it over the next 0.1 s (eval/sim_eval.py --interp lag)",
           "images": "re-rendered offline from recorded qpos (r2s/render.py, quality fast, 256x256 h264)"}
  t1 = time.perf_counter()
  info = write_dataset(out, eps, fps=10, video_keys=list(RENAME.values()), video_size=(256, 256),
                       video_codec="h264", robot_type="so101_follower", extra_meta=extra,
                       move_videos=True)
  write_s = time.perf_counter() - t1
  (out / "meta" / "kept_ids.json").write_text(json.dumps(
    {"select": a.select, "ids": [[r["seed"], r["attempt"]] for r in kept],
     "n_rollouts": len(rows), "n_kept": len(kept)}, indent=1))
  if not a.no_pretrained_norm:
    subprocess.run([sys.executable, str(ROOT / "datagen" / "use_pretrained_norm.py"), str(out)],
                   check=True, stdout=subprocess.DEVNULL)
  try:
    stage.rmdir()
  except OSError:
    pass
  print(json.dumps({"dataset": str(out), "episodes": info["total_episodes"], "frames": info["total_frames"],
                    "render_wall_s": round(render_wall, 1), "workers": a.workers,
                    "render_s_per_episode": round(float(np.mean([v["render_s"] for v in rend.values()])), 2),
                    "write_s": round(write_s, 1)}, indent=1))


if __name__ == "__main__":
  main()
