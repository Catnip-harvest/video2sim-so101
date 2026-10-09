"""Closed-loop simulation evaluation of an SO-101 pick-and-place policy (or of the scripted
expert / a recorded dataset episode, replayed through the very same path).

  # policy (LeRobot pretrained_model dir), 20 held-out seeds
  python eval/sim_eval.py --mode policy --checkpoint CKPT/pretrained_model --n 20 --out OUT
  # scripted expert, re-expressed as 10 Hz MolmoAct2-frame actions, through the harness
  python eval/sim_eval.py --mode expert --n 20 --out OUT
  # recorded actions of training episodes 0..4, open loop through the harness
  python eval/sim_eval.py --mode dataset-replay --episodes 0-4 --out OUT

One episode (all modes):
  1. Variation from (seed, attempt) with variety=True, exactly as datagen draws it. Eval
     seeds default to --seed 1000; the training set used --seed 1 (no overlap: the
     episode seed is seed*1_000_003 + attempt).
  2. Scene built and settled exactly as rollout.run_episode does (same builder, same
     initial placement, same 0.3 s settle). Nothing writes qpos/qvel after that.
  3. Every 10 Hz step (= 3 sim frames of 1/30 s = 21 physics steps):
       observation.state  = measured joints (sim rad) -> calib/so101_units: sim_to_lerobot
                            (default ArmCalib) -> lerobot_to_molmoact2, float32 [6]
       images             = cam_head (sim "front") and cam_right (sim "wrist_cam") rendered
                            like datagen: separate render model + MjData set to the current
                            qpos, render quality "fast", 256x256 uint8 -> float32 [0,1] CHW
                            (lerobot's prepare_observation_for_inference); only rendered when
                            the policy will read them (start of each 10-action chunk) or for
                            video
       task               = "pick up the cube and place it in the bin"
     action (MolmoAct2 frame) -> molmoact2_to_lerobot -> lerobot_to_sim (rad) -> clipped to
     the joint range -> expanded to 3 frame goals -> rate-limited to 98 % of the expert's
     servo caps (arm 0.5 rad/s, jaw 2 rad/s, planner.rate_limit) -> each frame's goal is
     ramped over its 7 physics steps (first-order hold, the expert's servo model).
     Expansion ("--interp"): lag (default) = the 3 frame goals ramp linearly from the last
     10 Hz action to the new one (causal, smooth, ~67 ms behind the expert's timing);
     hold = jump to the new action at once (the rate limiter then spreads it).
  4. Phase-free verifier (datagen/r2s/verify_policy.py) on every physics step; verdict at
     the end: pass (= L_full), L_task, reasons.
  Policy mode stops early once the task has been done for 1 s (cube resting in the bin,
  no jaw contact) or the cube fell off the table; otherwise at --max-sim-s.

Writes OUT/episodes.jsonl (one row per episode), OUT/summary.json (success rate with a 95 %
Wilson CI, throughput, clipping, inference latency) and OUT/videos/*.mp4 for --videos N.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa"  # EGL fails on this droplet (container default is egl)
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.setdefault("LP_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "datagen", ROOT / "calib", ROOT / "eval"):
  if str(p) not in sys.path:
    sys.path.insert(0, str(p))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from r2s import scene as S  # noqa: E402
from r2s.paths import load_task_spec  # noqa: E402
from r2s.planner import ARM_VMAX, JAW_VMAX  # noqa: E402
from r2s.render import _apply_quality, _Encoder  # noqa: E402
from r2s.rollout import SETTLE_S  # noqa: E402
from r2s.variations import Variation, sample_variation  # noqa: E402
from r2s.verify_policy import PhaseFreeMonitor, run_expert_judged  # noqa: E402
from so101_units import (ArmCalib, lerobot_to_molmoact2, lerobot_to_sim,  # noqa: E402
                         molmoact2_to_lerobot, sim_to_lerobot)

SPEC = ROOT / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"
DATASET = Path("/work/outputs_large/datagen/sugar_v10_n1000_molmoact2")
TASK = "pick up the cube and place it in the bin"
STEP_FRAMES = 3  # 30 Hz sim frames per 10 Hz policy step
CAMS = {"observation.images.cam_head": "front", "observation.images.cam_right": "wrist_cam"}
RATE_FRAC = 0.98  # same margin as planner.rate_limit


# ---- unit conversion (the one module: calib/so101_units.py) ------------------------------
def sim_to_mm2(q, cal=None) -> np.ndarray:
  return lerobot_to_molmoact2(sim_to_lerobot(q, cal)).astype(np.float32)


def mm2_to_sim(a, cal=None) -> np.ndarray:
  return lerobot_to_sim(molmoact2_to_lerobot(np.asarray(a, np.float64)), cal)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
  if n == 0:
    return (0.0, 1.0)
  p = k / n
  den = 1 + z * z / n
  c = (p + z * z / (2 * n)) / den
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return (max(0.0, c - h), min(1.0, c + h))


# ---- scene and observation -----------------------------------------------------------------
def setup_scene(spec, v: Variation, size=(256, 256), init_obj_z: float | None = None) -> S.EpisodeScene:
  """Same build, initial placement and settle as rollout.run_episode (the only state write).
  ``init_obj_z`` (tests only) starts the object at that centre height instead of on the
  table, e.g. resting on the bin floor for the "cube already in the bin" negative control."""
  sc = S.build_episode_scene(v, object_kind=spec.object.kind, render_size=size)
  m, d = sc.model, sc.data
  assert m.neq == 0 and m.nmocap == 0, "model must have no equality constraints or mocap"
  mujoco.mj_resetData(m, d)
  if init_obj_z is not None:
    d.qpos[sc.obj_qadr + 2] = init_obj_z
  q0 = np.array(v.q_start)
  d.qpos[sc.arm_qadr] = q0
  d.qpos[sc.g_qadr] = v.gripper_start
  d.ctrl[sc.arm_act] = q0
  d.ctrl[sc.g_act] = v.gripper_start
  mujoco.mj_forward(m, d)
  for _ in range(int(SETTLE_S / m.opt.timestep)):
    mujoco.mj_step(m, d)
  d.time = 0.0
  return sc


class ObsRenderer:
  """Renders like r2s/render.render_episode: its own model built from the Variation, the
  quality flags of --render-quality, and a separate MjData set to the live qpos (rendering
  never touches the physics)."""

  def __init__(self, spec, v: Variation, size=(256, 256), quality: str = "fast"):
    sc = S.build_episode_scene(v, object_kind=spec.object.kind, render_size=size)
    self.m = sc.model
    self.flags = _apply_quality(self.m, quality)
    self.d = mujoco.MjData(self.m)
    w, h = size
    self.r = mujoco.Renderer(self.m, h, w)

  def render(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
    self.d.qpos[:] = qpos
    mujoco.mj_forward(self.m, self.d)
    out = {}
    for key, cam in CAMS.items():
      self.r.update_scene(self.d, camera=cam)
      for f, val in self.flags.items():
        self.r.scene.flags[f] = val
      out[key] = self.r.render().copy()
    return out

  def close(self) -> None:
    self.r.close()


# ---- action sources ------------------------------------------------------------------------
class ReplaySource:
  """Feeds a fixed array of 10 Hz actions (MolmoAct2 frame) open loop."""
  needs_images = False

  def __init__(self, actions_mm2: np.ndarray):
    self.a = np.asarray(actions_mm2)

  def reset(self):
    pass

  def needs_obs(self) -> bool:
    return False

  def act(self, k: int, state, images):
    return None if k >= len(self.a) else self.a[k]


class PolicySource:
  needs_images = True

  def __init__(self, runner):
    self.runner = runner

  def reset(self):
    self.runner.reset()

  def needs_obs(self) -> bool:
    return self.runner.needs_obs()

  def act(self, k: int, state, images):
    return self.runner.act(state, images)


# ---- one episode ---------------------------------------------------------------------------
def run_harness_episode(spec, v: Variation, source, *, max_sim_s: float = 40.0,
                        interp: str = "lag", early_stop: bool = False, video: Path | None = None,
                        quality: str = "fast", size=(256, 256), tail_s: float = 0.0,
                        cal: ArmCalib | None = None, keep_obs: list | None = None,
                        init_obj_z: float | None = None) -> dict:
  t_wall = time.perf_counter()
  sc = setup_scene(spec, v, size, init_obj_z)
  m, d = sc.model, sc.data
  mon = PhaseFreeMonitor(sc)
  rend = ObsRenderer(spec, v, size, quality) if (source.needs_images or video or keep_obs is not None) else None
  enc = _Encoder(video, (2 * size[0], size[1]), "h264", fps=10) if video else None
  lo, hi = sc.joint_lo, sc.joint_hi
  cap = np.array([ARM_VMAX] * 5 + [JAW_VMAX]) * RATE_FRAC / S.FPS
  last_goal = np.r_[d.ctrl[sc.arm_act], d.ctrl[sc.g_act]].astype(float)  # 10 Hz goal
  last_frame = last_goal.copy()  # last applied 30 Hz frame goal
  n_steps = int(round(max_sim_s * 10))
  tm = {"render": 0.0, "policy": 0.0, "physics": 0.0}
  clip = {"frames": 0, "frames_clipped": 0, "steps": 0, "steps_range_clipped": 0,
          "max_excess_rad": 0.0}
  done_run, stop_reason, k = 0, "time_limit", 0
  source.reset()
  for k in range(n_steps):
    q = np.r_[d.qpos[sc.arm_qadr], d.qpos[sc.g_qadr]]
    state = sim_to_mm2(q, cal)
    imgs = None
    if rend is not None and (video or source.needs_obs() or keep_obs is not None):
      t0 = time.perf_counter()
      imgs = rend.render(d.qpos)
      tm["render"] += time.perf_counter() - t0
      if enc:
        enc.add(np.concatenate([imgs["observation.images.cam_head"],
                                imgs["observation.images.cam_right"]], axis=1))
      if keep_obs is not None:
        keep_obs.append({"k": k, "state": state, "images": imgs})
    t0 = time.perf_counter()
    a = source.act(k, state, imgs)
    tm["policy"] += time.perf_counter() - t0
    if a is None:
      stop_reason = "actions_exhausted"
      break
    goal = mm2_to_sim(a, cal)
    clip["steps"] += 1
    g2 = np.clip(goal, lo, hi)
    if np.any(np.abs(g2 - goal) > 1e-9):
      clip["steps_range_clipped"] += 1
    goal = g2
    if interp == "lag":
      frames = [last_goal + (j + 1) / STEP_FRAMES * (goal - last_goal) for j in range(STEP_FRAMES)]
    elif interp == "hold":
      frames = [goal] * STEP_FRAMES
    else:
      raise ValueError(interp)
    last_goal = goal
    t0 = time.perf_counter()
    for f in frames:
      dlt = f - last_frame
      lim = np.clip(dlt, -cap, cap)
      clip["frames"] += 1
      if np.any(np.abs(lim - dlt) > 1e-9):
        clip["frames_clipped"] += 1
        clip["max_excess_rad"] = max(clip["max_excess_rad"], float(np.max(np.abs(dlt - lim))))
      tgt = last_frame + lim
      mon.frame_start(tgt)
      prev = np.r_[d.ctrl[sc.arm_act], d.ctrl[sc.g_act]]
      for j in range(S.SUBSTEPS):  # first-order hold, as rollout._run_frames(servo_foh=True)
        al = (j + 1) / S.SUBSTEPS
        d.ctrl[sc.arm_act] = prev[:5] + al * (tgt[:5] - prev[:5])
        d.ctrl[sc.g_act] = prev[5] + al * (tgt[5] - prev[5])
        mujoco.mj_step(m, d)
        mon.step(v)
      last_frame = tgt
    tm["physics"] += time.perf_counter() - t0
    if mon.base.c["unstable"]:
      stop_reason = "unstable"
      break
    if early_stop:
      op = d.xpos[sc.obj_body]
      if op[2] < -0.02:
        stop_reason = "object_fell_off_table"
        break
      bx, by = v.bin_xy
      lim_xy = 0.055 - spec.place.success_xy_margin
      sp = float(np.linalg.norm(d.qvel[sc.obj_dadr:sc.obj_dadr + 3]))
      ok = (abs(op[0] - bx) <= lim_xy and abs(op[1] - by) <= lim_xy and sp < 0.02
            and d.time - mon.last_jaw_contact_t > 0.5 and not mon.carrying)
      done_run = done_run + 1 if ok else 0
      if done_run >= 10:
        stop_reason = "task_done_1s"
        break
  # Replays: hold the last goal a little so the cube settles (expert data ends in 0.5 s rest).
  t0 = time.perf_counter()
  for _ in range(int(round(tail_s * S.FPS))):
    mon.frame_start(last_frame)
    for _ in range(S.SUBSTEPS):
      mujoco.mj_step(m, d)
      mon.step(v)
  tm["physics"] += time.perf_counter() - t0
  if enc:
    enc.close()
  if rend:
    rend.close()
  verdict = mon.verdict(spec, v)
  return {"verdict": verdict, "steps_10hz": clip["steps"], "stop_reason": stop_reason,
          "sim_s": round(float(d.time), 3), "clip": clip,
          "timing_s": {k2: round(x, 3) for k2, x in tm.items()},
          "wall_s": round(time.perf_counter() - t_wall, 3)}


# ---- modes ---------------------------------------------------------------------------------
_G: dict = {}


def _init(cfg: dict) -> None:
  _G["cfg"] = cfg
  _G["spec"] = load_task_spec(cfg["spec"])
  if cfg["mode"] == "policy":
    from policy_io import PolicyRunner
    t0 = time.perf_counter()
    _G["runner"] = PolicyRunner(cfg["checkpoint"], device=cfg["device"], task=TASK)
    _G["load_s"] = round(time.perf_counter() - t0, 1)


def _dataset_episode(e: int) -> tuple[Variation, np.ndarray, np.ndarray]:
  import pyarrow.compute as pc
  import pyarrow.dataset as pads
  root = Path(_G["cfg"]["dataset"])
  meta = None
  with open(root / "meta" / "datagen_episodes.jsonl") as f:
    for line in f:
      r = json.loads(line)
      if r["episode_index"] == e:
        meta = r
        break
  assert meta is not None, f"episode {e} not in datagen_episodes.jsonl"
  ds = pads.dataset(str(root / "data"), format="parquet")
  t = ds.to_table(filter=pc.field("episode_index") == e,
                  columns=["frame_index", "action", "observation.state"]).sort_by("frame_index")
  act = np.array(t.column("action").to_pylist(), np.float32)
  st = np.array(t.column("observation.state").to_pylist(), np.float32)
  return Variation.from_dict(meta["variation"]), act, st


def _episode(item) -> dict:
  cfg, spec = _G["cfg"], _G["spec"]
  mode, idx = cfg["mode"], int(item)
  vid = None
  if idx in cfg.get("video_ids", []):
    vdir = Path(cfg["out"]) / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    vid = vdir / f"{mode}_{idx:05d}.mp4"
  row: dict = {"mode": mode}
  kw = dict(max_sim_s=cfg["max_sim_s"], interp=cfg["interp"], video=vid, quality=cfg["quality"])
  if mode == "policy":
    v = sample_variation(spec, cfg["seed"], idx, variety=True)
    r = run_harness_episode(spec, v, PolicySource(_G["runner"]), early_stop=True, **kw)
    run = _G["runner"]
    r["inference_ms"] = {"chunks": len(run.chunk_ms),
                         "chunk_median": round(float(np.median(run.chunk_ms)), 1) if run.chunk_ms else None,
                         "chunk_max": round(float(np.max(run.chunk_ms)), 1) if run.chunk_ms else None}
    run.chunk_ms.clear()
    run.call_ms.clear()
    row.update({"seed": cfg["seed"], "attempt": idx, "episode_seed": v.seed})
  elif mode == "expert":
    v = sample_variation(spec, cfg["seed"], idx, variety=True)
    t0 = time.perf_counter()
    res, pf = run_expert_judged(spec, v, pinch_width=cfg["pinch_width"])
    exp_s = time.perf_counter() - t0
    a10 = sim_to_mm2(res.action[::STEP_FRAMES]) if res.action is not None else np.zeros((0, 6), np.float32)
    r = run_harness_episode(spec, v, ReplaySource(a10), tail_s=cfg["tail_s"], **kw)
    row.update({"seed": cfg["seed"], "attempt": idx, "episode_seed": v.seed,
                "expert_native": {"expert_verifier_pass": bool(res.accepted),
                                  "expert_reasons": res.reasons,
                                  "phase_free_pass": bool(pf["passed"]), "phase_free_reasons": pf["reasons"],
                                  "frames_30hz": 0 if res.action is None else int(len(res.action)),
                                  "wall_s": round(exp_s, 2)}})
  elif mode == "dataset-replay":
    v, act, st = _dataset_episode(idx)
    keep = [] if cfg.get("check_images") else None
    r = run_harness_episode(spec, v, ReplaySource(act), tail_s=cfg["tail_s"], keep_obs=keep, **kw)
    # State the harness observed vs the state recorded in the dataset at the same step.
    row.update({"episode_index": idx, "attempt": v.attempt, "episode_seed": v.seed,
                "dataset_frames": int(len(act))})
    if keep:
      row["obs_check"] = _compare_with_dataset(idx, keep, st)
  else:
    raise ValueError(mode)
  vd = r.pop("verdict")
  row.update({"pass": vd["passed"], "L_task": vd["L_task"], "L_full": vd["L_full"],
              "reasons": vd["reasons"], "style_only": vd["style_only"], **r,
              "verify_stats": vd["stats"]})
  if vid:
    row["video"] = str(vid)
  return row


def _compare_with_dataset(e: int, keep: list, st: np.ndarray) -> dict:
  """Harness observation vs what LeRobotDataset gives the policy for the same episode step."""
  from lerobot.datasets.lerobot_dataset import LeRobotDataset
  ds = _G.get("lds")
  if ds is None:
    ds = _G["lds"] = LeRobotDataset("local/" + Path(_G["cfg"]["dataset"]).name,
                                    root=_G["cfg"]["dataset"], video_backend="pyav")
  start = int(ds.meta.episodes["dataset_from_index"][e])
  n = min(len(keep), len(st))
  ks = sorted({0, 1, 5, n // 4, n // 2, 3 * n // 4, n - 1})
  out = {"frames_checked": ks, "per_cam_mean_abs_diff_0_255": {}, "per_cam_psnr_db": {}}
  item0 = ds[start]
  out["dataset_image_dtype"] = str(item0["observation.images.cam_head"].dtype)
  out["dataset_image_shape"] = list(item0["observation.images.cam_head"].shape)
  out["dataset_image_range"] = [float(item0["observation.images.cam_head"].min()),
                                float(item0["observation.images.cam_head"].max())]
  for key in CAMS:
    mads, ps = [], []
    for k in ks:
      ref = (ds[start + k][key].numpy().transpose(1, 2, 0) * 255.0)
      img = keep[k]["images"][key].astype(np.float64)
      diff = img - ref
      mads.append(float(np.abs(diff).mean()))
      mse = float((diff ** 2).mean())
      ps.append(99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse))
    out["per_cam_mean_abs_diff_0_255"][key] = [round(x, 2) for x in mads]
    out["per_cam_psnr_db"][key] = [round(x, 1) for x in ps]
  hs = np.array([o["state"] for o in keep[:n]])
  dd = np.abs(hs - st[:n])
  out["state_abs_diff_deg"] = {"step0_max": round(float(dd[0].max()), 4),
                               "median": round(float(np.median(dd)), 3),
                               "p95": round(float(np.quantile(dd, 0.95)), 3),
                               "max": round(float(dd.max()), 3)}
  return out


def summarize(rows: list[dict], cfg: dict, wall_s: float, load_s: float | None) -> dict:
  n = len(rows)
  k_full = sum(r["L_full"] for r in rows)
  k_task = sum(r["L_task"] for r in rows)
  from collections import Counter
  reasons = Counter(x for r in rows for x in r["reasons"])
  frames = sum(r["clip"]["frames"] for r in rows)
  fclip = sum(r["clip"]["frames_clipped"] for r in rows)
  steps = sum(r["clip"]["steps"] for r in rows)
  rclip = sum(r["clip"]["steps_range_clipped"] for r in rows)
  ep_wall = [r["wall_s"] for r in rows]
  s = {
    "mode": cfg["mode"], "n": n, "seed": cfg.get("seed"), "checkpoint": cfg.get("checkpoint"),
    "interp": cfg["interp"], "max_sim_s": cfg["max_sim_s"], "workers": cfg["workers"],
    "success_L_full": k_full, "success_rate_L_full": round(k_full / max(n, 1), 4),
    "wilson95_L_full": [round(x, 4) for x in wilson(k_full, n)],
    "success_L_task": k_task, "success_rate_L_task": round(k_task / max(n, 1), 4),
    "wilson95_L_task": [round(x, 4) for x in wilson(k_task, n)],
    "style_only_failures": sum(r["style_only"] for r in rows),
    "reason_counts": dict(reasons.most_common()),
    "stop_reasons": dict(Counter(r["stop_reason"] for r in rows)),
    "rate_limit_clipped_frames": fclip, "frames": frames,
    "rate_limit_clipped_frac": round(fclip / max(frames, 1), 4),
    "joint_range_clipped_steps": rclip, "steps_10hz": steps,
    "max_rate_excess_rad": round(max([r["clip"]["max_excess_rad"] for r in rows] or [0.0]), 4),
    "wall_s_total": round(wall_s, 1), "episode_wall_s_mean": round(float(np.mean(ep_wall)), 2) if rows else None,
    "sim_s_mean": round(float(np.mean([r["sim_s"] for r in rows])), 2) if rows else None,
    "episodes_per_hour_wall": round(3600 * n / max(wall_s, 1e-9), 1),
    "timing_s_mean": {k: round(float(np.mean([r["timing_s"][k] for r in rows])), 2)
                      for k in ("render", "policy", "physics")} if rows else {},
    "policy_load_s": load_s,
  }
  if cfg["mode"] == "policy":
    ch = [r["inference_ms"]["chunk_median"] for r in rows if r["inference_ms"]["chunk_median"]]
    s["inference_chunk_ms_median_of_episode_medians"] = round(float(np.median(ch)), 1) if ch else None
  if cfg["mode"] == "expert":
    en = [r["expert_native"] for r in rows]
    s["expert_native_verifier_pass"] = sum(e["expert_verifier_pass"] for e in en)
    s["expert_native_phase_free_pass"] = sum(e["phase_free_pass"] for e in en)
    s["harness_vs_native_expert_verifier_agree"] = sum(
      e["expert_verifier_pass"] == r["L_full"] for e, r in zip(en, rows))
  return s


def _parse_ids(spec: str) -> list[int]:
  out = []
  for part in spec.split(","):
    if "-" in part:
      a, b = part.split("-")
      out += list(range(int(a), int(b) + 1))
    elif part:
      out.append(int(part))
  return out


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--mode", required=True, choices=["policy", "expert", "dataset-replay"])
  ap.add_argument("--policy-source", default=None, help="alias: 'dataset-replay' = --mode dataset-replay")
  ap.add_argument("--checkpoint", default=None, help="lerobot pretrained_model dir (policy mode)")
  ap.add_argument("--n", type=int, default=20)
  ap.add_argument("--start", type=int, default=0, help="first attempt id")
  ap.add_argument("--seed", type=int, default=1000, help="eval base seed (training used 1)")
  ap.add_argument("--episodes", default="0-4", help="dataset episode ids (dataset-replay)")
  ap.add_argument("--dataset", default=str(DATASET))
  ap.add_argument("--spec", default=str(SPEC))
  ap.add_argument("--max-sim-s", type=float, default=40.0)
  ap.add_argument("--interp", default="lag", choices=["lag", "hold"])
  ap.add_argument("--quality", default="fast", choices=["full", "shadow", "fast"])
  ap.add_argument("--tail-s", type=float, default=0.3, help="replay modes: hold the last goal this long")
  ap.add_argument("--videos", type=int, default=0, help="write mp4s for the first N episodes")
  ap.add_argument("--check-images", action="store_true",
                  help="dataset-replay: compare harness images/state with LeRobotDataset frames")
  ap.add_argument("--workers", type=int, default=1, help="processes (policy mode: one replica each)")
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  if a.policy_source == "dataset-replay":
    a.mode = "dataset-replay"
  if a.mode == "policy" and not a.checkpoint:
    ap.error("--checkpoint is required in policy mode")
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  from generate import pinch_width_from_spec
  ids = _parse_ids(a.episodes) if a.mode == "dataset-replay" else list(range(a.start, a.start + a.n))
  cfg = {"mode": a.mode, "checkpoint": a.checkpoint, "seed": a.seed, "dataset": a.dataset,
         "spec": a.spec, "max_sim_s": a.max_sim_s, "interp": a.interp, "quality": a.quality,
         "tail_s": a.tail_s, "video_ids": ids[: a.videos], "out": str(out),
         "pinch_width": pinch_width_from_spec(Path(a.spec)), "device": a.device,
         "workers": a.workers, "check_images": a.check_images, "ids": ids}
  (out / "config.json").write_text(json.dumps(cfg, indent=1))
  t0 = time.perf_counter()
  rows = []
  log = open(out / "episodes.jsonl", "w")

  def emit(r):
    rows.append(r)
    log.write(json.dumps(r) + "\n")
    log.flush()
    print(f"[eval {a.mode}] {len(rows)}/{len(ids)} id={r.get('attempt', r.get('episode_index'))} "
          f"pass={r['pass']} L_task={r['L_task']} reasons={r['reasons']} sim={r['sim_s']}s "
          f"wall={r['wall_s']}s stop={r['stop_reason']}", flush=True)

  load_s = None
  if a.workers <= 1:
    _init(cfg)
    load_s = _G.get("load_s")
    for i in ids:
      emit(_episode(i))
  else:
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
      for r in pool.imap_unordered(_episode, ids):
        emit(r)
  log.close()
  wall = time.perf_counter() - t0
  s = summarize(rows, cfg, wall, load_s)
  if load_s:
    s["episodes_per_hour_excluding_load"] = round(3600 * len(rows) / max(wall - load_s, 1e-9), 1)
  (out / "summary.json").write_text(json.dumps(s, indent=1))
  print(json.dumps(s, indent=1))


if __name__ == "__main__":
  main()
