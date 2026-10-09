"""Real-to-sim data engine CLI: one task spec -> N physically verified SO-101 episodes.

  python generate.py --spec SPEC.json --n 1000 --workers 16 --render --out runs/sugar

Steps per attempt (deterministic: attempt i always uses seed (seed, i)):
  sample variation -> plan (scripted expert) -> simulate with full physics -> verify
  -> log one line to <out>/attempts.jsonl -> if accepted, stage the arrays (and render).
When N attempts have been accepted, the first N accepted attempt ids (in id order, so the
result does not depend on --workers or scheduling) are written as a LeRobot v3.0 dataset
to <out>/lerobot, plus <out>/acceptance.json and <out>/acceptance.md.

Resumable: re-running the same command skips attempts already in attempts.jsonl.
--finalize-only rewrites the dataset from the staged episodes.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from r2s.paths import load_task_spec  # noqa: E402

DEFAULT_TASK = "pick up the cube and place it in the bin"
ROBOT_TYPE = "so101_follower"

_W: dict = {}  # per-worker globals


def pinch_width_from_spec(spec_path: Path) -> float | None:
  """The extractor's two-finger pinch width (gripper.grasp_width_m), if the so101 spec
  points at the extractor's own task_spec.json."""
  try:
    d = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    ext = Path(d.get("source", {}).get("extractor_spec", "") or "/nonexistent")
    if not ext.is_file():  # absolute Windows path on another machine: use the sibling file
      ext = Path(spec_path).with_name("task_spec.json")
    if ext.is_file():
      g = json.loads(ext.read_text(encoding="utf-8")).get("gripper", {})
      if "grasp_width_m" in g:
        return float(g["grasp_width_m"])
  except Exception:  # noqa: BLE001
    pass
  return None


def _init_worker(cfg: dict) -> None:
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  _W["cfg"] = cfg
  _W["spec"] = load_task_spec(cfg["spec"])


def _attempt(i: int) -> dict:
  from r2s.rollout import run_episode
  from r2s.variations import sample_variation

  cfg, spec = _W["cfg"], _W["spec"]
  try:
    v = sample_variation(spec, cfg["seed"], i)
    res = run_episode(spec, v, pinch_width=cfg["pinch_width"],
                      render_size=tuple(cfg["size"]), servo_foh=cfg["servo_hold"] == "foh")
  except Exception as e:  # noqa: BLE001  (logged as a rejection, never silently dropped)
    import traceback
    return {"attempt": i, "accepted": False, "reasons": ["exception"],
            "primary_reason": "exception", "frames": 0, "n_frames": 0, "stats": {},
            "plan": {"error": traceback.format_exc(limit=3)}, "variation": {"servo_variant": "?"},
            "sim_seconds": 0.0, "wall_seconds": 0.0}
  row = res.summary()
  row["n_frames"] = 0 if res.state is None else int(len(res.state))
  if res.accepted:
    stage = Path(cfg["out"]) / "staging"
    stage.mkdir(parents=True, exist_ok=True)
    stem = stage / f"attempt_{i:06d}"
    np.savez_compressed(str(stem) + ".npz", state=res.state, action=res.action,
                        env_state=res.env_state, qpos_full=res.qpos_full,
                        phases=np.array(res.phases))
    row["image_stats"] = {}
    if cfg["render"]:
      from r2s.render import render_episode
      t0 = time.perf_counter()
      paths = {k: f"{stem}.{k.split('.')[-1]}.mp4" for k in cfg["video_keys"]}
      row["image_stats"] = render_episode(v, res.qpos_full, cfg["video_keys"], paths,
                                          size=tuple(cfg["size"]), codec=cfg["codec"],
                                          object_kind=spec.object.kind,
                                          quality=cfg.get("render_quality", "full"))
      row["render_seconds"] = round(time.perf_counter() - t0, 3)
    (Path(str(stem) + ".json")).write_text(json.dumps(row))
  return row


def load_log(out: Path) -> dict[int, dict]:
  rows = {}
  p = out / "attempts.jsonl"
  if p.exists():
    for line in p.read_text(encoding="utf-8").splitlines():
      if line.strip():
        r = json.loads(line)
        rows[int(r["attempt"])] = r
  return rows


def staged_ok(out: Path, i: int, cfg: dict) -> bool:
  stem = out / "staging" / f"attempt_{i:06d}"
  if not (Path(str(stem) + ".npz").exists() and Path(str(stem) + ".json").exists()):
    return False
  if cfg["render"]:
    return all(Path(f"{stem}.{k.split('.')[-1]}.mp4").exists() for k in cfg["video_keys"])
  return True


def acceptance_report(rows: list[dict], wall_s: float, workers: int, cfg: dict) -> dict:
  n = len(rows)
  acc = [r for r in rows if r["accepted"]]
  prim = Counter(r["primary_reason"] for r in rows if not r["accepted"])
  anyr = Counter(x for r in rows if not r["accepted"] for x in r["reasons"])
  cpu = sum(r["wall_seconds"] + r.get("render_seconds", 0.0) for r in rows)
  sim = sum(r["sim_seconds"] for r in rows)
  slip = [r["stats"].get("in_hand_slip_mm") for r in acc if r["stats"].get("in_hand_slip_mm") is not None]
  tilt = [r["plan"]["release"]["object_tilt_in_hand_deg"] for r in acc
          if r.get("plan", {}).get("release", {}).get("object_tilt_in_hand_deg") is not None]
  place = [np.hypot(*r["stats"]["final_obj_offset_from_bin_mm"]) for r in acc
           if "final_obj_offset_from_bin_mm" in r["stats"]]
  by_servo = {}
  for variant in ("7v4", "12v"):
    sub = [r for r in rows if r["variation"]["servo_variant"] == variant]
    if sub:
      by_servo[variant] = {"attempts": len(sub),
                           "accepted": sum(r["accepted"] for r in sub),
                           "rate": round(sum(r["accepted"] for r in sub) / len(sub), 4)}
  return {
    "spec": cfg["spec"], "seed": cfg["seed"], "attempts": n, "accepted": len(acc),
    "acceptance_rate": round(len(acc) / max(n, 1), 4),
    "rejections_by_primary_reason": dict(prim.most_common()),
    "rejections_any_reason": dict(anyr.most_common()),
    "acceptance_by_servo_variant": by_servo,
    "cpu_seconds_total": round(cpu, 1), "sim_seconds_total": round(sim, 1),
    "attempts_per_cpu_second": round(n / max(cpu, 1e-9), 4),
    "accepted_episodes_per_cpu_second": round(len(acc) / max(cpu, 1e-9), 4),
    "sim_to_cpu_ratio": round(sim / max(cpu, 1e-9), 2),
    "wall_seconds_this_session": round(wall_s, 1), "workers": workers,
    "render": cfg["render"],
    "accepted_mean_frames": round(float(np.mean([r["n_frames"] for r in acc])), 1) if acc else 0,
    "accepted_in_hand_slip_mm": _q(slip), "accepted_final_offset_from_bin_centre_mm": _q(place),
    "accepted_object_tilt_in_hand_before_lowering_deg": _q(tilt),
  }


def _q(x):
  if not x:
    return None
  x = np.asarray(x, float)
  return {"median": round(float(np.median(x)), 2), "p90": round(float(np.quantile(x, 0.9)), 2),
          "max": round(float(x.max()), 2)}


def write_report_md(rep: dict, path: Path) -> None:
  lines = [f"# Acceptance: {Path(rep['spec']).parent.name}", "",
           f"- attempts {rep['attempts']}, accepted {rep['accepted']} "
           f"(**{100 * rep['acceptance_rate']:.1f} %**)",
           f"- {rep['attempts_per_cpu_second']} attempts / CPU-second, "
           f"{rep['accepted_episodes_per_cpu_second']} accepted episodes / CPU-second "
           f"(render={rep['render']})",
           f"- accepted episodes: mean {rep['accepted_mean_frames']} frames at 30 Hz; "
           f"in-hand slip {rep['accepted_in_hand_slip_mm']} mm; object tilt in hand before "
           f"lowering {rep['accepted_object_tilt_in_hand_before_lowering_deg']} deg; final offset "
           f"from bin centre {rep['accepted_final_offset_from_bin_centre_mm']} mm",
           f"- by servo: {rep['acceptance_by_servo_variant']}", "",
           "| primary rejection reason | count | % of attempts |", "|---|---:|---:|"]
  for k, c in rep["rejections_by_primary_reason"].items():
    lines.append(f"| {k} | {c} | {100 * c / max(rep['attempts'], 1):.1f} |")
  lines += ["", "| any rejection reason (an attempt can have several) | count |", "|---|---:|"]
  for k, c in rep["rejections_any_reason"].items():
    lines.append(f"| {k} | {c} |")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _phase_starts(phases) -> dict:
  out = {}
  for k, p in enumerate(phases.tolist()):
    out.setdefault(str(p), k)
  return out


def finalize(out: Path, cfg: dict, rows: dict[int, dict], wall_s: float, workers: int) -> None:
  from r2s.lerobot_io import EpisodeData, write_dataset

  acc_ids = sorted(i for i, r in rows.items() if r["accepted"])[: cfg["n"]]
  last = acc_ids[-1] if acc_ids else -1
  # Report over the deterministic prefix of attempts (ids <= last selected one).
  considered = [rows[i] for i in sorted(rows) if i <= last] if acc_ids else list(rows.values())
  rep = acceptance_report(considered, wall_s, workers, cfg)
  rep["attempts_logged_total"] = len(rows)
  (out / "acceptance.json").write_text(json.dumps(rep, indent=2))
  write_report_md(rep, out / "acceptance.md")
  print(json.dumps({k: rep[k] for k in ("attempts", "accepted", "acceptance_rate",
                                        "rejections_by_primary_reason")}, indent=1))
  if not acc_ids:
    print("no accepted episodes; dataset not written")
    return
  if cfg.get("no_dataset"):
    return
  eps = []
  for i in acc_ids:
    stem = out / "staging" / f"attempt_{i:06d}"
    z = np.load(str(stem) + ".npz")
    meta = json.loads(Path(str(stem) + ".json").read_text())
    videos = {k: Path(f"{stem}.{k.split('.')[-1]}.mp4") for k in cfg["video_keys"]} \
      if cfg["render"] else {}
    eps.append(EpisodeData(
      state=z["state"], action=z["action"], env_state=z["env_state"], task=cfg["task"],
      videos=videos, image_stats=meta.get("image_stats", {}) or {},
      meta={"attempt": i, "seed": meta["variation"]["seed"], "variation": meta["variation"],
            "verification": meta["stats"], "plan": meta["plan"],
            "phase_starts": _phase_starts(z["phases"])}))
  root = out / "lerobot"
  if root.exists():
    shutil.rmtree(root)
  extra = {"generator": "manip/datagen/generate.py", "config": cfg, "acceptance": rep,
           "units": "joint angles in radians (MuJoCo), object pose in metres / wxyz quaternion, "
                    "robot base frame",
           "action_semantics": "absolute joint position targets held for 1/30 s"}
  info = write_dataset(root, eps, fps=30, video_keys=cfg["video_keys"] if cfg["render"] else [],
                       video_size=tuple(cfg["size"]), video_codec=cfg["codec"],
                       robot_type=ROBOT_TYPE, extra_meta=extra, move_videos=False)
  print(f"wrote {root}: {info['total_episodes']} episodes, {info['total_frames']} frames")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--spec", required=True)
  ap.add_argument("--n", type=int, default=1000, help="accepted episodes wanted")
  ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
  ap.add_argument("--out", required=True)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--render", dest="render", action="store_true")
  ap.add_argument("--no-render", dest="render", action="store_false")
  ap.set_defaults(render=False)
  ap.add_argument("--size", default="256x256", help="WxH of rendered images")
  ap.add_argument("--render-quality", default="full", choices=["full", "shadow", "fast"],
                  help="full: shadows+reflection+MSAA (slow on OSMesa); shadow: 1024 shadow map, "
                       "no reflection/MSAA; fast: none of them")
  ap.add_argument("--no-wrist", action="store_true", help="front camera only")
  ap.add_argument("--codec", default="h264", choices=["h264", "av1"])
  ap.add_argument("--task", default=DEFAULT_TASK)
  ap.add_argument("--servo-hold", default="foh", choices=["foh", "zoh"],
                  help="foh: servo goal ramps over each 33 ms frame (default); "
                       "zoh: goal jumps once per frame (ablation)")
  ap.add_argument("--max-attempts", type=int, default=None,
                  help="stop after this many attempts (default 4*n + 50)")
  ap.add_argument("--finalize-only", action="store_true")
  ap.add_argument("--no-dataset", action="store_true",
                  help="log + acceptance report only (no LeRobot dataset)")
  args = ap.parse_args()

  out = Path(args.out).resolve()
  out.mkdir(parents=True, exist_ok=True)
  w, h = (int(x) for x in args.size.lower().split("x"))
  spec_path = str(Path(args.spec).resolve())
  load_task_spec(spec_path)  # validate early
  cfg = {
    "spec": spec_path, "n": args.n, "seed": args.seed, "render": args.render, "size": [w, h],
    "codec": args.codec, "task": args.task, "out": str(out),
    "video_keys": ["observation.images.front"] + ([] if args.no_wrist else
                                                  ["observation.images.wrist"]),
    "pinch_width": pinch_width_from_spec(Path(spec_path)), "no_dataset": args.no_dataset,
    "servo_hold": args.servo_hold, "render_quality": args.render_quality,
  }
  run_json = out / "run.json"
  if run_json.exists():
    old = json.loads(run_json.read_text())
    for k in ("spec", "seed", "size", "codec", "video_keys", "pinch_width", "servo_hold",
              "render_quality"):
      if old.get(k, "full" if k == "render_quality" else None) != cfg[k]:
        sys.exit(f"{out} was started with {k}={old.get(k)!r}, now {cfg[k]!r}; use a new --out")
    if old.get("render") and not cfg["render"]:
      sys.exit(f"{out} was started with --render; keep --render to resume")
  run_json.write_text(json.dumps(cfg, indent=2))

  if args.render:
    from r2s.render import gl_available
    ok, msg = gl_available()
    if not ok:
      sys.exit(f"--render needs OpenGL (MUJOCO_GL=egl or osmesa on Linux): {msg}")

  rows = load_log(out)
  # Accepted attempts whose staged files are missing (e.g. killed mid-write, or an
  # earlier no-render run being resumed with --render) are redone.
  redo = [i for i, r in rows.items() if r["accepted"] and not staged_ok(out, i, cfg)]
  for i in redo:
    rows.pop(i)
  max_attempts = args.max_attempts or 4 * args.n + 50
  t0 = time.perf_counter()
  if not args.finalize_only:
    def accepted_ids():
      return sorted(i for i, r in rows.items() if r["accepted"])

    def need_more() -> bool:
      acc = accepted_ids()
      if len(acc) < args.n:
        return len(rows) < max_attempts
      cutoff = acc[args.n - 1]  # every id below the N-th accepted one must be done
      return any(i not in rows for i in range(cutoff))

    def todo():
      i = 0
      while True:
        if i not in rows:
          yield i
        i += 1

    log = open(out / "attempts.jsonl", "a", encoding="utf-8")
    ctx = mp.get_context("spawn")
    inflight_max = max(2, 2 * args.workers)
    print(f"[datagen] spec={Path(spec_path).parent.name} n={args.n} workers={args.workers} "
          f"render={args.render} resume: {len(rows)} logged, "
          f"{len(accepted_ids())} accepted", flush=True)
    done_now = 0
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
      gen = todo()
      pending = {}
      while need_more() or pending:
        while need_more() and len(pending) < inflight_max:
          i = next(gen)
          if i in pending:
            continue
          acc = accepted_ids()
          if len(acc) >= args.n and i > acc[args.n - 1]:
            break
          pending[i] = pool.apply_async(_attempt, (i,))
        ready = [i for i, r in pending.items() if r.ready()]
        if not ready:
          time.sleep(0.05)
          continue
        for i in ready:
          row = pending.pop(i).get()
          rows[i] = row
          slim = {k: row[k] for k in row if k != "image_stats"}
          log.write(json.dumps(slim) + "\n")
          log.flush()
          done_now += 1
          if done_now % 10 == 0 or row["accepted"] and len(accepted_ids()) % 50 == 0:
            el = time.perf_counter() - t0
            acc_n = len(accepted_ids())
            print(f"[datagen] {len(rows)} attempts, {acc_n} accepted "
                  f"({100 * acc_n / max(len(rows), 1):.1f} %), {done_now / el:.2f} attempts/s "
                  f"wall", flush=True)
    log.close()
  finalize(out, cfg, rows, time.perf_counter() - t0, args.workers)


if __name__ == "__main__":
  main()
