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
     --ood <rung> (eval/ood.py, docs/OOD_LADDER.md) then edits that Variation for an
     out-of-distribution test; --ood none (default) leaves it untouched.
  2. Scene built and settled exactly as rollout.run_episode does (same builder, same
     initial placement, same 0.3 s settle). Nothing writes qpos/qvel after that.
  3. Every 10 Hz step (= 3 sim frames of 1/30 s = 21 physics steps):
       observation.state  = measured joints (sim rad) -> calib/so101_units: sim_to_lerobot
                            (default ArmCalib) -> lerobot_to_molmoact2, float32 [6]
       images             = cam_head (sim "front") and cam_right (sim "wrist_cam") rendered
                            like datagen: separate render model + MjData set to the current
                            qpos, render quality "fast", 256x256 uint8, then (default
                            --obs-codec h264) one encode/decode with the dataset's video
                            settings, since training frames were h264-decoded -> float32
                            [0,1] CHW (lerobot's prepare_observation_for_inference, same
                            dtype/range/shape LeRobotDataset gives); only rendered when
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

Writes OUT/episodes.jsonl (one row per episode), OUT/actions/*.npz (the 10 Hz actions and
observed states, so any episode can be replayed/re-judged on CPU), OUT/summary.json (success rate with a 95 %
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
from ood import RUNGS as OOD_RUNGS, apply_exposure, apply_ood, exposure_of  # noqa: E402

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


def h264_roundtrip(img: np.ndarray) -> np.ndarray:
  """Encode one frame with the datagen video settings (libx264 crf 23, veryfast, yuv420p)
  and decode it, so the policy sees the same codec artefacts its training frames had."""
  import io

  import av
  buf = io.BytesIO()
  c = av.open(buf, mode="w", format="mp4")
  st = c.add_stream("libx264", rate=10)
  st.width, st.height = img.shape[1], img.shape[0]
  st.pix_fmt = "yuv420p"
  st.options = {"crf": "23", "g": "2", "preset": "veryfast"}
  f = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
  for pk in st.encode(f):
    c.mux(pk)
  for pk in st.encode():
    c.mux(pk)
  c.close()
  buf.seek(0)
  with av.open(buf) as c2:
    return next(iter(c2.decode(video=0))).to_ndarray(format="rgb24")


class ObsRenderer:
  """Renders like r2s/render.render_episode: its own model built from the Variation, the
  quality flags of --render-quality, and a separate MjData set to the live qpos (rendering
  never touches the physics)."""

  def __init__(self, spec, v: Variation, size=(256, 256), quality: str = "fast",
               codec: bool = False):
    self.codec = codec
    self.exposure = exposure_of(v)  # --ood exposure: image transform "in the sensor"
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
      img = self.r.render().copy()
      if self.exposure:
        img = apply_exposure(img, **self.exposure)
      out[key] = h264_roundtrip(img) if self.codec else img
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


class EMASource:
  """Control B (a): exponential moving average of the 10 Hz actions,
  out_k = alpha * a_k + (1 - alpha) * out_{k-1}, out_0 = a_0. All 6 joints."""

  def __init__(self, inner, alpha: float):
    self.inner, self.alpha = inner, float(alpha)
    self.needs_images = inner.needs_images
    self.prev = None

  def reset(self):
    self.inner.reset()
    self.prev = None

  def needs_obs(self) -> bool:
    return self.inner.needs_obs()

  def act(self, k: int, state, images):
    a = self.inner.act(k, state, images)
    if a is None:
      return None
    a = np.asarray(a, np.float64)
    self.prev = a if self.prev is None else self.alpha * a + (1 - self.alpha) * self.prev
    return self.prev.copy()


class BlendSource:
  """Control B (b): re-plan every ``every`` steps (< chunk length) and cross-fade the
  overlapping part of the old and new chunks: on the overlap the new chunk's weight rises
  linearly 1/(n+1) .. n/(n+1). ``runner`` needs ``chunk(state, images) -> (H, 6)`` (MolmoAct2
  frame). Costs chunk/every times the inference of plain select_action (2x for 10/5)."""
  needs_images = True

  def __init__(self, runner, every: int = 5):
    self.runner, self.every = runner, int(every)
    self.plan: np.ndarray | None = None  # actions for steps start .. start+H-1
    self.start = 0

  def reset(self):
    if hasattr(self.runner, "reset"):
      self.runner.reset()
    self.plan, self.start = None, 0

  def needs_obs(self) -> bool:
    return True  # the caller's step counter decides; see act()

  def wants_obs(self, k: int) -> bool:
    return self.plan is None or k - self.start >= self.every

  def act(self, k: int, state, images):
    if self.wants_obs(k):
      new = np.asarray(self.runner.chunk(state, images, k=k), np.float64)
      if self.plan is not None:
        old = self.plan[k - self.start:]
        n = min(len(old), len(new))
        w = (np.arange(1, n + 1) / (n + 1))[:, None]
        new = new.copy()
        new[:n] = (1 - w) * old[:n] + w * new[:n]
      self.plan, self.start = new, k
    i = k - self.start
    if i >= len(self.plan):  # cannot happen with every <= H
      i = len(self.plan) - 1
    return self.plan[i].copy()


# ---- one episode ---------------------------------------------------------------------------
def run_harness_episode(spec, v: Variation, source, *, max_sim_s: float = 40.0,
                        interp: str = "lag", early_stop: bool = False, video: Path | None = None,
                        quality: str = "fast", size=(256, 256), tail_s: float = 0.0,
                        cal: ArmCalib | None = None, keep_obs: list | None = None,
                        init_obj_z: float | None = None, obs_codec: bool = False) -> dict:
  t_wall = time.perf_counter()
  sc = setup_scene(spec, v, size, init_obj_z)
  m, d = sc.model, sc.data
  mon = PhaseFreeMonitor(sc)
  rend = ObsRenderer(spec, v, size, quality, obs_codec) if (source.needs_images or video or keep_obs is not None) else None
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
  log_a: list = []  # actions as the source gave them (MolmoAct2 frame), for CPU re-judging
  log_s: list = []
  log_g: list = []  # goal actually commanded (joint-range clipped), MolmoAct2 frame
  log_env: list = []  # object pos + quat (wxyz) at the step start, like datagen's env_state
  log_q: list = []  # full qpos at the step start (offline re-render, like datagen)
  source.reset()
  for k in range(n_steps):
    q = np.r_[d.qpos[sc.arm_qadr], d.qpos[sc.g_qadr]]
    state = sim_to_mm2(q, cal)
    imgs = None
    want = source.wants_obs(k) if hasattr(source, "wants_obs") else source.needs_obs()
    if rend is not None and (video or want or keep_obs is not None):
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
    log_a.append(np.asarray(a, np.float32))
    log_s.append(state)
    log_env.append(np.r_[d.xpos[sc.obj_body], d.xquat[sc.obj_body]].astype(np.float32))
    log_q.append(d.qpos.copy())
    goal = mm2_to_sim(a, cal)
    clip["steps"] += 1
    g2 = np.clip(goal, lo, hi)
    if np.any(np.abs(g2 - goal) > 1e-9):
      clip["steps_range_clipped"] += 1
    goal = g2
    log_g.append(sim_to_mm2(goal, cal))
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
  return {"verdict": verdict, "actions_mm2": np.array(log_a, np.float32).reshape(-1, 6),
          "states_mm2": np.array(log_s, np.float32).reshape(-1, 6),
          "goals_mm2": np.array(log_g, np.float32).reshape(-1, 6),
          "env_state": np.array(log_env, np.float32).reshape(-1, 7),
          "qpos_full": np.array(log_q, np.float64).reshape(len(log_q), -1),
          "steps_10hz": clip["steps"], "stop_reason": stop_reason,
          "sim_s": round(float(d.time), 3), "clip": clip,
          "timing_s": {k2: round(x, 3) for k2, x in tm.items()},
          "wall_s": round(time.perf_counter() - t_wall, 3)}


# ---- modes ---------------------------------------------------------------------------------
_G: dict = {}


def _exit_with_parent() -> None:
  """Pool workers (each holding a GPU replica) exit if the parent process dies."""
  import threading
  ppid = os.getppid()

  def watch():
    while True:
      time.sleep(2)
      if os.getppid() != ppid:
        os._exit(1)
  threading.Thread(target=watch, daemon=True).start()


def _init(cfg: dict, worker: bool = False) -> None:
  if worker:
    _exit_with_parent()
  _G["cfg"] = cfg
  _G["spec"] = load_task_spec(cfg["spec"])
  if cfg["mode"] == "policy":
    from policy_io import PolicyRunner
    t0 = time.perf_counter()
    _G["runner"] = PolicyRunner(cfg["checkpoint"], device=cfg["device"], task=TASK)
    _G["load_s"] = round(time.perf_counter() - t0, 1)


class ExpertChunkRunner:
  """Stand-in for a policy's chunk API in CPU tests of BlendSource: returns the expert's own
  10 Hz actions for steps k .. k+H-1 (last action repeated past the end)."""

  def __init__(self, a10: np.ndarray, horizon: int = 10):
    self.a, self.h = np.asarray(a10), horizon

  def reset(self):
    pass

  def chunk(self, state, images, k: int = 0):
    idx = np.minimum(np.arange(k, k + self.h), len(self.a) - 1)
    return self.a[idx]


class ObsEvery:
  """Replay modes: also render (+ codec) the observation every ``every`` steps, as policy
  mode does at chunk starts, to measure the CPU side of a policy eval without a GPU."""

  def __init__(self, inner, every: int):
    self.inner, self.every = inner, int(every)
    self.needs_images = True

  def reset(self):
    self.inner.reset()

  def needs_obs(self) -> bool:
    return False

  def wants_obs(self, k: int) -> bool:
    return k % self.every == 0

  def act(self, k, state, images):
    return self.inner.act(k, state, images)


def _smoothed(base, chunk_runner, cfg):
  """Wrap an action source with control-B smoothing (--smooth)."""
  sm = cfg.get("smooth", "none")
  if sm == "none":
    return base
  if sm == "ema":
    return EMASource(base, cfg["ema_alpha"])
  if sm == "blend":
    if isinstance(base, ReplaySource):  # CPU stand-in: replay ends when the expert's actions do
      src = BlendSource(chunk_runner, cfg["replan_every"])
      n = len(base.a)
      inner = src.act
      src.act = lambda k, st, im: None if k >= n else inner(k, st, im)
      src.needs_images = False
      return src
    return BlendSource(chunk_runner, cfg["replan_every"])
  raise ValueError(sm)


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
  row: dict = {"mode": mode, "ood": cfg.get("ood", "none")}
  kw = dict(max_sim_s=cfg["max_sim_s"], interp=cfg["interp"], video=vid, quality=cfg["quality"],
            obs_codec=cfg["obs_codec"])
  if mode == "policy":
    v = apply_ood(sample_variation(spec, cfg["seed"], idx, variety=True), cfg.get("ood", "none"))
    run = _G["runner"]
    r = run_harness_episode(spec, v, _smoothed(PolicySource(run), run, cfg), early_stop=True, **kw)
    r["inference_ms"] = {"chunks": len(run.chunk_ms),
                         "chunk_median": round(float(np.median(run.chunk_ms)), 1) if run.chunk_ms else None,
                         "chunk_max": round(float(np.max(run.chunk_ms)), 1) if run.chunk_ms else None}
    run.chunk_ms.clear()
    run.call_ms.clear()
    row.update({"seed": cfg["seed"], "attempt": idx, "episode_seed": v.seed})
  elif mode == "expert":
    v = apply_ood(sample_variation(spec, cfg["seed"], idx, variety=True), cfg.get("ood", "none"))
    t0 = time.perf_counter()
    res, pf = run_expert_judged(spec, v, pinch_width=cfg["pinch_width"])
    exp_s = time.perf_counter() - t0
    a10 = sim_to_mm2(res.action[::STEP_FRAMES]) if res.action is not None else np.zeros((0, 6), np.float32)
    src = _smoothed(ReplaySource(a10), ExpertChunkRunner(a10), cfg)
    if cfg.get("render_obs_every"):
      src = ObsEvery(src, cfg["render_obs_every"])
    r = run_harness_episode(spec, v, src, tail_s=cfg["tail_s"], **kw)
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
  rdir = Path(cfg["out"]) / "rollouts"
  rdir.mkdir(parents=True, exist_ok=True)
  # Everything needed to (a) replay/re-judge on CPU: ReplaySource(action_raw_mm2) reproduces
  # the episode (same scene and physics; the source's actions are fixed once chosen), and (b)
  # write a training dataset like datagen's (eval/rollouts_to_dataset.py): per 10 Hz step the
  # observed state, the commanded goal, the object pose and the full qpos for re-rendering.
  np.savez_compressed(rdir / f"{mode}_{idx:05d}.npz", action_raw_mm2=r.pop("actions_mm2"),
                      state_mm2=r.pop("states_mm2"), action_mm2=r.pop("goals_mm2"),
                      env_state=r.pop("env_state"), qpos_full=r.pop("qpos_full"))
  row["rollout_file"] = str(rdir / f"{mode}_{idx:05d}.npz")
  if mode != "dataset-replay":
    row["variation"] = v.to_dict()
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
  # Same frames through the datagen video encoder (h264 crf 23, render._Encoder) and back:
  # isolates codec noise. Frame 0 shows an identical scene state in both (state diff 0).
  import tempfile

  import av
  m0 = min(n, 12)
  out["codec_roundtrip_frame0_psnr_db"] = {}
  for key in CAMS:
    with tempfile.TemporaryDirectory() as td:
      fp = Path(td) / "x.mp4"
      enc = _Encoder(fp, keep[0]["images"][key].shape[1::-1], "h264", fps=10)
      for o in keep[:m0]:
        enc.add(o["images"][key])
      enc.close()
      with av.open(str(fp)) as c:
        dec0 = next(iter(c.decode(video=0))).to_ndarray(format="rgb24").astype(np.float64)
    ref = ds[start][key].numpy().transpose(1, 2, 0) * 255.0
    mse = float(((dec0 - ref) ** 2).mean())
    out["codec_roundtrip_frame0_psnr_db"][key] = 99.0 if mse == 0 else round(10 * math.log10(255 ** 2 / mse), 1)
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
    "obs_codec": "h264" if cfg.get("obs_codec") else "none",
    "smooth": cfg.get("smooth", "none"), "ema_alpha": cfg.get("ema_alpha"),
    "replan_every": cfg.get("replan_every"), "ood": cfg.get("ood", "none"),
    "wrong_object_in_bin": sum("wrong_object_in_bin" in r["reasons"] for r in rows),
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
    # steady state (policy load excluded): K processes each spending mean episode wall time
    "episodes_per_hour_steady": round(3600 * cfg["workers"] / max(float(np.mean(ep_wall)), 1e-9), 1)
    if rows else None,
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
  ap.add_argument("--obs-codec", default="h264", choices=["h264", "none"],
                  help="pass each observation image through the dataset's h264 settings "
                       "(training frames were h264-decoded); none = raw renders")
  ap.add_argument("--videos", type=int, default=0, help="write mp4s for the first N episodes")
  ap.add_argument("--check-images", action="store_true",
                  help="dataset-replay: compare harness images/state with LeRobotDataset frames")
  ap.add_argument("--workers", type=int, default=1, help="processes (policy mode: one replica each)")
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--smooth", default="none", choices=["none", "ema", "blend"],
                  help="control B: ema = EMA of 10 Hz actions; blend = re-plan every "
                       "--replan-every steps and cross-fade overlapping chunks (2x inference)")
  ap.add_argument("--ema-alpha", type=float, default=0.5)
  ap.add_argument("--replan-every", type=int, default=5)
  ap.add_argument("--render-obs-every", type=int, default=0,
                  help="expert mode: render+codec the observation every N steps like a policy "
                       "would (CPU-cost measurement); 0 = off")
  ap.add_argument("--ood", default="none", choices=list(OOD_RUNGS),
                  help="out-of-distribution rung (docs/OOD_LADDER.md); edits each Variation "
                       "after sample_variation. none = in-range scenes, unchanged")
  ap.add_argument("--fresh", action="store_true", help="ignore an existing episodes.jsonl in --out")
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
         "workers": a.workers, "check_images": a.check_images, "ids": ids,
         "obs_codec": a.obs_codec == "h264", "smooth": a.smooth, "ema_alpha": a.ema_alpha,
         "replan_every": a.replan_every, "render_obs_every": a.render_obs_every, "ood": a.ood}
  # Resumable: rows already in OUT/episodes.jsonl are kept and their ids skipped, provided
  # the run-defining settings are the same.
  rows = []
  ep_log = out / "episodes.jsonl"
  if ep_log.exists() and not a.fresh:
    old = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    for k in ("mode", "checkpoint", "seed", "spec", "max_sim_s", "interp", "quality", "tail_s",
              "obs_codec", "smooth", "ema_alpha", "replan_every", "ood"):
      if k in old and old[k] != cfg[k]:
        sys.exit(f"{out} holds a run with {k}={old[k]!r}, now {cfg[k]!r}: use another --out or --fresh")
    for line in ep_log.read_text().splitlines():
      if line.strip():
        rows.append(json.loads(line))
    done = {r.get("attempt", r.get("episode_index")) for r in rows}
    ids = [i for i in ids if i not in done]
    cfg["ids"] = sorted(done | set(ids))
    print(f"[eval] resume: {len(rows)} episodes already in {ep_log}, {len(ids)} to run", flush=True)
  (out / "config.json").write_text(json.dumps(cfg, indent=1))
  t0 = time.perf_counter()
  n_total = len(rows) + len(ids)
  log = open(ep_log, "w" if a.fresh else "a")

  def emit(r):
    rows.append(r)
    log.write(json.dumps(r) + "\n")
    log.flush()
    print(f"[eval {a.mode}] {len(rows)}/{n_total} id={r.get('attempt', r.get('episode_index'))} "
          f"pass={r['pass']} L_task={r['L_task']} reasons={r['reasons']} sim={r['sim_s']}s "
          f"wall={r['wall_s']}s stop={r['stop_reason']}", flush=True)

  load_s = None
  if not ids:
    pass
  elif a.workers <= 1:
    _init(cfg)
    load_s = _G.get("load_s")
    for i in ids:
      emit(_episode(i))
  else:
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(cfg, True)) as pool:
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
