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

--hanoi switches the task to one Tower-of-Hanoi move per episode (r2s/hanoi.py): a random
legal non-goal start state, the scripted expert makes the solver's optimal next move, the
Hanoi verifier (r2s/hanoi_verify.py) judges it and only L_full episodes are kept. Every
episode has the same instruction (hanoi.INSTRUCTION); its start state, move and expected
end state are in meta/datagen_episodes.jsonl ("hanoi" and variation.extras.hanoi).
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


def record_step(cfg: dict) -> int:
  """Keep every k-th 30 Hz sim frame (state, action and video alike): 30 Hz -> k=1, 10 Hz -> 3.
  Same rule as calib/so101_units.subsample, which is applied to the real episodes."""
  hz = int(cfg.get("record_hz", 30))
  if 30 % hz:
    raise ValueError(f"--record-hz must divide 30, got {hz}")
  return 30 // hz


def _init_worker(cfg: dict) -> None:
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  _W["cfg"] = cfg
  _W["spec"] = load_task_spec(cfg["spec"])


def _attempt(i: int) -> dict:
  from r2s.hanoi_expert import run_hanoi_episode
  from r2s.rollout import run_episode
  cfg, spec = _W["cfg"], _W["spec"]
  try:
    v = variation_for(spec, cfg, i)
    runner = run_hanoi_episode if cfg.get("hanoi") else run_episode
    res = runner(spec, v, pinch_width=cfg["pinch_width"],
                 render_size=tuple(cfg["size"]), servo_foh=cfg["servo_hold"] == "foh")
  except Exception as e:  # noqa: BLE001  (logged as a rejection, never silently dropped)
    import traceback
    return {"attempt": i, "accepted": False, "reasons": ["exception"],
            "primary_reason": "exception", "frames": 0, "n_frames": 0, "stats": {},
            "plan": {"error": traceback.format_exc(limit=3)}, "variation": {"servo_variant": "?"},
            "sim_seconds": 0.0, "wall_seconds": 0.0}
  row = res.summary()
  row["n_frames"] = 0 if res.state is None else int(len(res.state))
  # Rejected attempts are staged and rendered too with --keep-rejects (negatives for the
  # JEPA judge); finalize() only ever reads accepted ones into the dataset.
  if res.accepted or (cfg.get("keep_rejects") and res.qpos_full is not None):
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
      step = record_step(cfg)
      row["image_stats"] = render_episode(v, res.qpos_full[::step], cfg["video_keys"], paths,
                                          size=tuple(cfg["size"]), codec=cfg["codec"],
                                          object_kind=spec.object.kind,
                                          quality=cfg.get("render_quality", "full"),
                                          fps=cfg.get("record_hz", 30))
      row["render_seconds"] = round(time.perf_counter() - t0, 3)
    (Path(str(stem) + ".json")).write_text(json.dumps(row))
  return row


def variation_for(spec, cfg: dict, i: int):
  """Attempt i's Variation: the single-cube one, (--multitask) the three-cube one, or
  (--hanoi) one Tower-of-Hanoi move."""
  from r2s.variations import sample_multitask_variation, sample_variation
  if cfg.get("hanoi"):
    from r2s.hanoi import sample_hanoi_variation
    return sample_hanoi_variation(spec, cfg["seed"], i, variety=cfg.get("variety", False))
  if cfg.get("multitask"):
    return sample_multitask_variation(spec, cfg["seed"], i, variety=cfg.get("variety", False),
                                      layout=cfg.get("mt_layout", "default"))
  return sample_variation(spec, cfg["seed"], i, variety=cfg.get("variety", False))


def episode_task(row: dict, default: str) -> str:
  """Per-episode task string: the multitask / Hanoi instruction if there is one, else --task."""
  extras = (row.get("variation") or {}).get("extras", {})
  mt = extras.get("multitask") or extras.get("hanoi")
  return mt["instruction"] if mt else default


def hanoi_meta(row: dict) -> dict | None:
  """Start state, move and expected end state of a Hanoi episode (None otherwise)."""
  h = (row.get("variation") or {}).get("extras", {}).get("hanoi")
  if not h:
    return None
  return {k: h[k] for k in ("start_state", "move", "end_state", "moved_cube", "spots")}


def target_of(row: dict) -> str | None:
  mt = (row.get("variation") or {}).get("extras", {}).get("multitask")
  return mt["target_colour"] if mt else None


def select_accepted(rows: dict[int, dict], n: int, quota: int | None = None) -> list[int]:
  """The episodes that go into the dataset: accepted attempt ids in id order (so the result
  does not depend on --workers), the first ``n``; with ``quota`` (multitask), the first
  ``quota`` per target colour, capped at ``n`` in total."""
  acc = sorted(i for i, r in rows.items() if r["accepted"])
  if quota is None:
    return acc[:n]
  out, cnt = [], Counter()
  for i in acc:
    t = target_of(rows[i])
    if cnt[t] < quota:
      cnt[t] += 1
      out.append(i)
      if len(out) >= n:
        break
  return out


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

  acc_ids = select_accepted(rows, cfg["n"], cfg.get("per_target"))
  last = max(acc_ids) if acc_ids else -1
  # Report over the deterministic prefix of attempts (ids <= last selected one).
  considered = [rows[i] for i in sorted(rows) if i <= last] if acc_ids else list(rows.values())
  rep = acceptance_report(considered, wall_s, workers, cfg)
  rep["attempts_logged_total"] = len(rows)
  if cfg.get("hanoi"):
    def cube_of(r):
      return (hanoi_meta(r) or {}).get("moved_cube", "?")
    h_att = Counter(cube_of(r) for r in considered)
    h_acc = Counter(cube_of(r) for r in considered if r["accepted"])
    rep["hanoi_acceptance_by_moved_cube"] = {c: {"attempts": h_att[c], "accepted": h_acc[c]}
                                             for c in sorted(h_att)}
    rep["hanoi_episodes_per_start_state"] = dict(sorted(Counter(
      hanoi_meta(rows[i])["start_state"] for i in acc_ids).items()))
    rep["hanoi_L_task_of_attempts"] = sum(bool(r["stats"].get("hanoi", {}).get("L_task"))
                                          for r in considered)
  if cfg.get("multitask"):
    rep["episodes_per_target"] = dict(Counter(target_of(rows[i]) for i in acc_ids))
    rep["episodes_per_task_string"] = dict(Counter(episode_task(rows[i], "") for i in acc_ids))
    mt_att = Counter(target_of(r) for r in considered)
    mt_acc = Counter(target_of(r) for r in considered if r["accepted"])
    rep["acceptance_by_target"] = {t: {"attempts": mt_att[t], "accepted": mt_acc[t],
                                       "rate": round(mt_acc[t] / max(mt_att[t], 1), 4)}
                                   for t in sorted(mt_att)}
    (out / "acceptance.json").write_text(json.dumps(rep, indent=2))
  (out / "acceptance.json").write_text(json.dumps(rep, indent=2))
  write_report_md(rep, out / "acceptance.md")
  print(json.dumps({k: rep[k] for k in ("attempts", "accepted", "acceptance_rate",
                                        "rejections_by_primary_reason")}, indent=1))
  if not acc_ids:
    print("no accepted episodes; dataset not written")
    return
  if cfg.get("no_dataset"):
    return
  export = cfg.get("export", {}) or {}
  frame = export.get("frame", "sim")
  cam_map = export.get("camera_names", {}) or {}  # e.g. {"front": "cam_head"}
  vkeys = cfg["video_keys"] if cfg["render"] else []
  renamed = {k: "observation.images." + cam_map.get(k.split(".")[-1], k.split(".")[-1]) for k in vkeys}
  convert, cal = None, None
  if frame != "sim":
    sys.path.insert(0, str(HERE.parent / "calib"))
    from so101_units import ArmCalib, lerobot_to_molmoact2, sim_to_lerobot
    cal = ArmCalib.load(export["arm_calib"]) if export.get("arm_calib") else ArmCalib()

    def convert(x):  # sim radians -> lerobot degrees (-> molmoact2 frame)
      y = sim_to_lerobot(x, cal)
      return (lerobot_to_molmoact2(y) if frame == "molmoact2" else y).astype(np.float32)
  eps = []
  for i in acc_ids:
    stem = out / "staging" / f"attempt_{i:06d}"
    z = dict(np.load(str(stem) + ".npz"))
    k = record_step(cfg)
    for key in ("state", "action", "env_state", "phases"):
      z[key] = z[key][::k]
    meta = json.loads(Path(str(stem) + ".json").read_text())
    videos = {renamed[k]: Path(f"{stem}.{k.split('.')[-1]}.mp4") for k in vkeys}
    if convert is not None:
      z["state"], z["action"] = convert(z["state"]), convert(z["action"])
    eps.append(EpisodeData(
      state=z["state"], action=z["action"], env_state=z["env_state"],
      task=episode_task(meta, cfg["task"]),
      videos=videos, image_stats=meta.get("image_stats", {}) or {},
      meta={"attempt": i, "seed": meta["variation"]["seed"], "variation": meta["variation"],
            "verification": meta["stats"], "plan": meta["plan"],
            "phase_starts": _phase_starts(z["phases"]),
            **({"hanoi": hanoi_meta(meta)} if hanoi_meta(meta) else {})}))
  root = Path(export["dataset_dir"]) if export.get("dataset_dir") else out / "lerobot"
  if root.exists():
    shutil.rmtree(root)
  extra = {"generator": "manip/datagen/generate.py", "config": cfg, "acceptance": rep,
           "units": {"sim": "joint angles in radians (MuJoCo Menagerie so101)",
                     "lerobot": "LeRobot new calibration (PR #777, use_degrees): degrees, "
                                "mid-range zero; gripper 0..100",
                     "molmoact2": "MolmoAct2-SO100_101 frame = LeRobot pre-PR #777: "
                                  "lift_old = 90 - lift_new, elbow_old = elbow_new + 90"}[frame]
                    + "; object pose in metres / wxyz quaternion, robot base frame",
           "joint_frame": frame, "arm_calib": None if cal is None else vars(cal),
           "conversion": "calib/so101_units.py",
           "action_semantics": f"absolute joint position target sent at this frame (sim runs "
                               f"30 Hz; recorded every {record_step(cfg)} sim frame(s))"}
  info = write_dataset(root, eps, fps=int(cfg.get("record_hz", 30)), video_keys=list(renamed.values()),
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
  ap.add_argument("--variety", action="store_true",
                  help="randomise the expert's tempo, heights, tilt, pauses and start pose")
  ap.add_argument("--record-hz", type=int, default=30, choices=[30, 15, 10],
                  help="dataset frame rate; the sim always runs at 30 Hz")
  ap.add_argument("--keep-rejects", action="store_true",
                  help="also stage and render rejected attempts (judge negatives)")
  ap.add_argument("--export-frame", default="sim", choices=["sim", "lerobot", "molmoact2"],
                  help="joint units/convention of the written dataset (calib/so101_units.py)")
  ap.add_argument("--arm-calib", default=None, help="ArmCalib JSON for sim<->lerobot (default: "
                  "the module's unverified defaults)")
  ap.add_argument("--camera-names", default="",
                  help="rename cameras in the dataset, e.g. front=cam_head,wrist=cam_right")
  ap.add_argument("--dataset-dir", default=None, help="write the dataset here (default <out>/lerobot)")
  ap.add_argument("--no-wrist", action="store_true", help="front camera only")
  ap.add_argument("--codec", default="h264", choices=["h264", "av1"])
  ap.add_argument("--task", default=DEFAULT_TASK)
  ap.add_argument("--multitask", action="store_true",
                  help="P3-A: three cubes, the instruction names one (per-episode task string)")
  ap.add_argument("--mt-layout", default="default", choices=["default", "narrow"])
  ap.add_argument("--hanoi", action="store_true",
                  help="Tower of Hanoi: one optimal move per episode from a random legal "
                       "start state (r2s/hanoi.py); constant instruction")
  ap.add_argument("--per-target", type=int, default=None,
                  help="multitask: accepted episodes per target colour (default n/3)")
  ap.add_argument("--servo-hold", default="foh", choices=["foh", "zoh"],
                  help="foh: servo goal ramps over each 33 ms frame (default); "
                       "zoh: goal jumps once per frame (ablation)")
  ap.add_argument("--max-attempts", type=int, default=None,
                  help="stop after this many attempts (default 4*n + 50)")
  ap.add_argument("--finalize-only", action="store_true")
  ap.add_argument("--no-dataset", action="store_true",
                  help="log + acceptance report only (no LeRobot dataset)")
  args = ap.parse_args()
  if args.hanoi and args.multitask:
    ap.error("--hanoi and --multitask are separate tasks")
  if args.hanoi and args.task == DEFAULT_TASK:
    from r2s.hanoi import INSTRUCTION
    args.task = INSTRUCTION

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
    "variety": args.variety, "record_hz": args.record_hz, "keep_rejects": args.keep_rejects,
    "multitask": args.multitask, "mt_layout": args.mt_layout,
    "per_target": (args.per_target or -(-args.n // 3)) if args.multitask else None,
    "hanoi": args.hanoi,
  }
  run_json = out / "run.json"
  if run_json.exists():
    old = json.loads(run_json.read_text())
    defaults = {"render_quality": "full", "variety": False, "record_hz": 30, "multitask": False,
                "mt_layout": "default", "hanoi": False}
    for k in ("spec", "seed", "size", "codec", "video_keys", "pinch_width", "servo_hold",
              "render_quality", "variety", "record_hz", "multitask", "mt_layout", "hanoi"):
      if old.get(k, defaults.get(k)) != cfg[k]:
        sys.exit(f"{out} was started with {k}={old.get(k)!r}, now {cfg[k]!r}; use a new --out")
    if old.get("render") and not cfg["render"]:
      sys.exit(f"{out} was started with --render; keep --render to resume")
  run_json.write_text(json.dumps(cfg, indent=2))
  # Export options only change how finalize() writes the dataset, so they are kept out of
  # run.json (a run can be re-exported with --finalize-only in another frame).
  cfg["export"] = {"frame": args.export_frame, "arm_calib": args.arm_calib,
                   "dataset_dir": args.dataset_dir,
                   "camera_names": dict(kv.split("=") for kv in args.camera_names.split(",") if kv)}

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

    def selected():
      return select_accepted(rows, args.n, cfg.get("per_target"))

    def need_more() -> bool:
      sel = selected()
      if len(sel) < args.n:
        return len(rows) < max_attempts
      cutoff = max(sel)  # every id below the last selected one must be done
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
          sel = selected()
          if len(sel) >= args.n and i > max(sel):
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
