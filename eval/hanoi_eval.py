"""Closed-loop Tower-of-Hanoi evaluation, run through sim_eval.py:

  # (a) one move per episode, from random legal non-goal start states (the training task)
  python eval/sim_eval.py --hanoi move --mode policy --checkpoint CKPT/pretrained_model --n 50 --out OUT
  python eval/sim_eval.py --hanoi move --mode expert --n 20 --out OUT
  # (b) the whole puzzle from "AAA": up to --max-moves move-episodes in ONE continuous scene
  python eval/sim_eval.py --hanoi puzzle --mode policy --checkpoint CKPT/pretrained_model --n 10 --out OUT
  python eval/sim_eval.py --hanoi puzzle --mode expert --n 5 --out OUT

Stepping is sim_eval.run_harness_episode's, unchanged: every 10 Hz step the policy gets the
measured joints (MolmoAct2 frame) and, at chunk starts, the cam_head / cam_right renders;
its action goes back through the unit conversion, the joint-range clip, the 3-frame "lag"
expansion, the 98 % servo-speed rate limit and the 7-substep first-order hold. The
instruction is always hanoi.INSTRUCTION. Judging is the Hanoi verifier
(datagen/r2s/hanoi_verify.py), on every physics step.

(a) move. Scene = sample_hanoi_variation(seed, id) (the datagen draw; eval seed 1000 vs.
    training seed 1 by default, so no overlap). Policy mode stops early once a move is
    done (some cube now rests on a different spot, the arrangement is readable and legal,
    every cube and the arm at rest, no jaw contact for 0.5 s, for 1 s in a row; the still
    arm means the next puzzle move-episode starts from rest), when a cube falls off the
    table, or at --max-sim-s. Expert mode runs the scripted expert natively (exactly
    as datagen does), then replays its actions at 10 Hz through the harness, like
    sim_eval's pick-and-place expert mode. Each row: L_task / L_full / reasons for the
    expected next state, plus the observed end state.
(b) puzzle. Scene = sample_hanoi_variation(seed, id, start_state="AAA"). Loop: read the
    state from the simulation; stop at the goal "CCC" or after --max-moves
    move-episodes; otherwise run one move-episode (policy: reset, run up to --move-max-s
    with the same early stop as (a); expert: plan the solver's next move on a COPY of the
    live simulation and replay those actions at 10 Hz in the live one). Nothing is reset
    between move-episodes. Each row: solved, moves used, illegal states reached (an
    arrangement with a larger cube on a smaller one, or a cube on no spot / not lying
    flat on its stack, after a move-episode), and every move-episode's verdict against the
    solver's move from the state it started in.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from pathlib import Path

import mujoco
import numpy as np

import sim_eval as E
from r2s import hanoi as H
from r2s import scene as S
from r2s.hanoi_expert import (TAIL_S, MoveSim, expert_move, place_and_settle,
                              run_hanoi_episode, run_hanoi_puzzle_expert)
from r2s.hanoi_verify import HanoiMonitor, read_state
from r2s.paths import rm
from r2s.planner import ARM_VMAX, JAW_VMAX, plan_config_for
from r2s.render import _Encoder

DONE_STEPS = 10  # 10 Hz steps in a row with the move done = 1 s
ARM_STILL = 0.05  # rad/s: every arm and jaw joint slower than this counts as still
FELL_Z = -0.02


class HanoiHarness:
  """One continuous scene driven by 10 Hz MolmoAct2-frame actions (policy or replay)."""

  def __init__(self, spec, v, size=(256, 256), quality: str = "fast", obs_codec: bool = False,
               need_images: bool = False, video: Path | None = None, cal=None):
    self.spec, self.v, self.cal = spec, v, cal
    self.sc = S.build_episode_scene(v, object_kind="box", render_size=size)
    place_and_settle(self.sc, v)
    self.mon = HanoiMonitor(self.sc, v)
    self.rend = (E.ObsRenderer(spec, v, size, quality, obs_codec)
                 if (need_images or video) else None)
    self.enc = _Encoder(video, (2 * size[0], size[1]), "h264", fps=10) if video else None
    sc, d = self.sc, self.sc.data
    self.lo, self.hi = sc.joint_lo, sc.joint_hi
    jids = [mujoco.mj_name2id(sc.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in rm.ALL_JOINTS]
    self.arm_dofs = np.array([sc.model.jnt_dofadr[j] for j in jids])
    self.cap = np.array([ARM_VMAX] * 5 + [JAW_VMAX]) * E.RATE_FRAC / S.FPS
    self.last_goal = np.r_[d.ctrl[sc.arm_act], d.ctrl[sc.g_act]].astype(float)
    self.last_frame = self.last_goal.copy()
    self.tm = {"render": 0.0, "policy": 0.0, "physics": 0.0}
    self.clip = {"frames": 0, "frames_clipped": 0, "steps": 0, "steps_range_clipped": 0,
                 "max_excess_rad": 0.0}
    self.log = {"action_raw_mm2": [], "state_mm2": [], "action_mm2": [], "cubes": [],
                "qpos_full": []}

  def new_segment(self) -> None:
    """A fresh verifier for the next move-episode (rest poses = the cubes as they are)."""
    self.mon = HanoiMonitor(self.sc, self.v)

  def positions(self) -> dict[str, np.ndarray]:
    return self.mon.positions()

  def state_mm2(self) -> np.ndarray:
    sc, d = self.sc, self.sc.data
    return E.sim_to_mm2(np.r_[d.qpos[sc.arm_qadr], d.qpos[sc.g_qadr]], self.cal)

  def _cubes_pose(self) -> np.ndarray:
    d = self.sc.data
    return np.array([np.r_[d.xpos[b], d.xquat[b]] for b in (self.mon.ids[c] for c in H.CUBES)],
                    np.float32)

  def step(self, a_mm2, interp: str = "lag") -> None:
    """One 10 Hz action, exactly as sim_eval.run_harness_episode applies it."""
    sc, d, m = self.sc, self.sc.data, self.sc.model
    goal = E.mm2_to_sim(a_mm2, self.cal)
    self.clip["steps"] += 1
    g2 = np.clip(goal, self.lo, self.hi)
    if np.any(np.abs(g2 - goal) > 1e-9):
      self.clip["steps_range_clipped"] += 1
    goal = g2
    self.log["action_mm2"].append(E.sim_to_mm2(goal, self.cal))
    if interp == "lag":
      frames = [self.last_goal + (j + 1) / E.STEP_FRAMES * (goal - self.last_goal)
                for j in range(E.STEP_FRAMES)]
    elif interp == "hold":
      frames = [goal] * E.STEP_FRAMES
    else:
      raise ValueError(interp)
    self.last_goal = goal
    t0 = time.perf_counter()
    for f in frames:
      dlt = f - self.last_frame
      lim = np.clip(dlt, -self.cap, self.cap)
      self.clip["frames"] += 1
      if np.any(np.abs(lim - dlt) > 1e-9):
        self.clip["frames_clipped"] += 1
        self.clip["max_excess_rad"] = max(self.clip["max_excess_rad"], float(np.max(np.abs(dlt - lim))))
      tgt = self.last_frame + lim
      self.mon.frame_start(tgt)
      prev = np.r_[d.ctrl[sc.arm_act], d.ctrl[sc.g_act]]
      for j in range(S.SUBSTEPS):
        al = (j + 1) / S.SUBSTEPS
        d.ctrl[sc.arm_act] = prev[:5] + al * (tgt[:5] - prev[:5])
        d.ctrl[sc.g_act] = prev[5] + al * (tgt[5] - prev[5])
        mujoco.mj_step(m, d)
        self.mon.step()
      self.last_frame = tgt
    self.tm["physics"] += time.perf_counter() - t0

  def hold(self, seconds: float) -> None:
    """Keep the last frame goal (replays end with a short settle, like sim_eval)."""
    t0 = time.perf_counter()
    for _ in range(int(round(seconds * S.FPS))):
      self.mon.frame_start(self.last_frame)
      for _ in range(S.SUBSTEPS):
        mujoco.mj_step(self.sc.model, self.sc.data)
        self.mon.step()
    self.tm["physics"] += time.perf_counter() - t0

  def move_done(self, start_state: str) -> bool:
    """A move has been made and everything has come to rest: some cube is on a new spot,
    the arrangement is readable and legal, no cube moves, no jaw has touched a cube for
    0.5 s, and the arm itself is still (so the next move-episode starts from rest)."""
    mon, d, sc = self.mon, self.sc.data, self.sc
    if any(mon.carrying.values()) or d.time - mon.last_jaw_contact_t <= 0.5 or not mon.at_rest():
      return False
    if float(np.max(np.abs(d.qvel[self.arm_dofs]))) > ARM_STILL:
      return False
    seen = read_state(self.v, self.positions())
    return seen["state"] is not None and seen["heights_ok"] and seen["state"] != start_state

  def run_segment(self, source, max_s: float, start_state: str, early_stop: bool,
                  interp: str = "lag", tail_s: float = 0.0) -> tuple[str, int]:
    """Drive the scene with ``source`` for one move-episode. Returns (stop reason, steps)."""
    d = self.sc.data
    t_start = float(d.time)
    done_run, stop, k = 0, "time_limit", 0
    source.reset()
    for k in range(int(round(max_s * 10))):
      state = self.state_mm2()
      imgs = None
      want = source.wants_obs(k) if hasattr(source, "wants_obs") else source.needs_obs()
      if self.rend is not None and (self.enc is not None or want):
        t0 = time.perf_counter()
        imgs = self.rend.render(d.qpos)
        self.tm["render"] += time.perf_counter() - t0
        if self.enc is not None:
          self.enc.add(np.concatenate([imgs["observation.images.cam_head"],
                                       imgs["observation.images.cam_right"]], axis=1))
      t0 = time.perf_counter()
      a = source.act(k, state, imgs)
      self.tm["policy"] += time.perf_counter() - t0
      if a is None:
        stop = "actions_exhausted"
        break
      self.log["action_raw_mm2"].append(np.asarray(a, np.float32))
      self.log["state_mm2"].append(state)
      self.log["cubes"].append(self._cubes_pose())
      self.log["qpos_full"].append(d.qpos.copy())
      self.step(a, interp)
      if self.mon.base.c["unstable"]:
        stop = "unstable"
        break
      if early_stop:
        if any(self.positions()[c][2] < FELL_Z for c in H.CUBES):
          stop = "cube_fell_off_table"
          break
        done_run = done_run + 1 if self.move_done(start_state) else 0
        if done_run >= DONE_STEPS:
          stop = "move_done_1s"
          break
    if tail_s > 0:
      self.hold(tail_s)
    return stop, round(float(d.time) - t_start, 3)

  def arrays(self) -> dict:
    out = {}
    for k, x in self.log.items():
      arr = np.array(x)
      out[k] = arr.astype(np.float64 if k == "qpos_full" else np.float32)
    return out

  def close(self) -> None:
    if self.enc is not None:
      self.enc.close()
    if self.rend is not None:
      self.rend.close()


def _expert_actions_on_copy(h: HanoiHarness, spec, state: str, move, pinch_width) -> tuple[np.ndarray, list]:
  """Plan and run the scripted expert's move on a COPY of the live simulation (the live
  scene is not touched); return its 10 Hz actions in the MolmoAct2 frame."""
  data = copy.copy(h.sc.data)
  sc2 = dataclasses.replace(h.sc, data=data)
  sim = MoveSim(sc2, h.v)
  info, fails = expert_move(sim, spec, h.v, state, tuple(move), plan_config_for(h.v), pinch_width,
                            start_target=h.last_frame)
  if sim.rec.action and not sim.unstable():
    sim.hold(TAIL_S)
  a30 = np.array(sim.rec.action, float).reshape(-1, 6)
  return E.sim_to_mm2(a30[::E.STEP_FRAMES], h.cal), fails


def _verdict_row(vd: dict) -> dict:
  return {"L_task": vd["L_task"], "L_full": vd["L_full"], "reasons": vd["reasons"],
          "style_only": vd["style_only"], "observed_end_state": vd["observed_end_state"],
          "expected_end_state": vd["expected_end_state"], "moved_cube": vd["moved_cube"]}


def _save(h: HanoiHarness, cfg: dict, tag: str, mover: str | None) -> str:
  rdir = Path(cfg["out"]) / "rollouts"
  rdir.mkdir(parents=True, exist_ok=True)
  arr = h.arrays()
  cubes = arr.pop("cubes").reshape(-1, len(H.CUBES), 7)
  env = cubes[:, H.CUBES.index(mover)] if mover else np.zeros((len(cubes), 7), np.float32)
  path = rdir / f"{tag}.npz"
  np.savez_compressed(path, env_state=env, cubes_state=cubes, cube_order=np.array(H.CUBES), **arr)
  return str(path)


def _source(cfg: dict, G: dict):
  run = G["runner"]
  return E._smoothed(E.PolicySource(run), run, cfg), run


def move_episode(idx: int, cfg: dict, spec, G: dict) -> dict:
  t_wall = time.perf_counter()
  mode = cfg["mode"]
  v = H.sample_hanoi_variation(spec, cfg["seed"], idx, variety=True)
  h_info = H.hanoi_info(v)
  vid = None
  if str(idx) in cfg.get("video_ids", []):
    vdir = Path(cfg["out"]) / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    vid = vdir / f"hanoi_move_{mode}_{idx:05d}.mp4"
  row: dict = {"mode": mode, "hanoi": "move", "run_key": str(idx), "seed": cfg["seed"],
               "attempt": idx, "episode_seed": v.seed, "instruction": H.INSTRUCTION,
               "start_state": h_info["start_state"], "move": h_info["move"]}
  h = HanoiHarness(spec, v, quality=cfg["quality"], obs_codec=cfg["obs_codec"],
                   need_images=(mode == "policy"), video=vid)
  if mode == "policy":
    src, run = _source(cfg, G)
    stop, seg_s = h.run_segment(src, cfg["max_sim_s"], h_info["start_state"], early_stop=True,
                                interp=cfg["interp"])
    row["inference_ms"] = {"chunks": len(run.chunk_ms),
                           "chunk_median": round(float(np.median(run.chunk_ms)), 1) if run.chunk_ms else None,
                           "chunk_max": round(float(np.max(run.chunk_ms)), 1) if run.chunk_ms else None}
    run.chunk_ms.clear()
    run.call_ms.clear()
  elif mode == "expert":
    t0 = time.perf_counter()
    res = run_hanoi_episode(spec, v, pinch_width=cfg["pinch_width"])
    a10 = (E.sim_to_mm2(res.action[::E.STEP_FRAMES]) if res.action is not None
           else np.zeros((0, 6), np.float32))
    row["expert_native"] = {"expert_verifier_pass": bool(res.accepted), "expert_reasons": res.reasons,
                            "phase_free_pass": bool(res.stats["hanoi"]["L_full"]),
                            "phase_free_reasons": res.stats["hanoi"]["reasons"],
                            "frames_30hz": 0 if res.action is None else int(len(res.action)),
                            "wall_s": round(time.perf_counter() - t0, 2)}
    stop, seg_s = h.run_segment(E.ReplaySource(a10), 1e6, h_info["start_state"], early_stop=False,
                                interp=cfg["interp"], tail_s=cfg["tail_s"])
  else:
    raise ValueError(f"--hanoi move supports --mode policy|expert, not {mode}")
  vd = h.mon.verdict(h_info["end_state"], h_info["moved_cube"])
  h.close()
  row["rollout_file"] = _save(h, cfg, f"hanoi_move_{mode}_{idx:05d}", h_info["moved_cube"])
  row.update({"pass": vd["passed"], **_verdict_row(vd), "stop_reason": stop,
              "sim_s": round(float(h.sc.data.time), 3), "steps_10hz": h.clip["steps"],
              "clip": h.clip, "timing_s": {k: round(x, 3) for k, x in h.tm.items()},
              "verify_stats": vd["stats"], "variation": v.to_dict(),
              "wall_s": round(time.perf_counter() - t_wall, 3)})
  if vid:
    row["video"] = str(vid)
  return row


def puzzle_episode(idx: int, cfg: dict, spec, G: dict) -> dict:
  t_wall = time.perf_counter()
  mode = cfg["mode"]
  if mode not in ("policy", "expert"):
    raise ValueError(f"--hanoi puzzle supports --mode policy|expert, not {mode}")
  v = H.sample_hanoi_variation(spec, cfg["seed"], idx, variety=True, start_state=H.START_STATE)
  vid = None
  if str(idx) in cfg.get("video_ids", []):
    vdir = Path(cfg["out"]) / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    vid = vdir / f"hanoi_puzzle_{mode}_{idx:05d}.mp4"
  h = HanoiHarness(spec, v, quality=cfg["quality"], obs_codec=cfg["obs_codec"],
                   need_images=(mode == "policy"), video=vid)
  seen = read_state(v, h.positions())
  state = seen["state"] if seen["heights_ok"] else None
  segments, illegal = [], 0
  stop_reason = "max_moves"
  for k in range(cfg["max_moves"]):
    if state == H.GOAL_STATE:
      stop_reason = "solved"
      break
    move = H.next_move(state) if state is not None else None
    expected = H.apply_move(state, move) if move else None
    h.new_segment()
    seg: dict = {"k": k, "from_state": state, "expected_move": list(move) if move else None,
                 "expected_state": expected}
    if mode == "policy":
      src, run = _source(cfg, G)
      stop, seg_s = h.run_segment(src, cfg["move_max_s"], state or "", early_stop=True,
                                  interp=cfg["interp"])
      seg["inference_chunks"] = len(run.chunk_ms)
      run.chunk_ms.clear()
      run.call_ms.clear()
    else:
      if move is None:
        stop_reason = "expert_cannot_read_state"
        break
      a10, fails = _expert_actions_on_copy(h, spec, state, move, cfg["pinch_width"])
      seg["expert_planner_failures"] = fails
      stop, seg_s = h.run_segment(E.ReplaySource(a10), 1e6, state, early_stop=False,
                                  interp=cfg["interp"], tail_s=cfg["tail_s"])
    seen = read_state(v, h.positions())
    bad = (not seen["legal"]) or seen["state"] is None or not seen["heights_ok"]
    illegal += bool(bad)
    seg.update({"stop_reason": stop, "seg_sim_s": seg_s, "observed_state": seen["state"],
                "observed_nearest": seen["nearest"], "legal": seen["legal"],
                "heights_ok": seen["heights_ok"], "off_spot_mm": seen["off_spot_mm"]})
    if move:
      vd = h.mon.verdict(expected, H.moved_cube(state, move))
      seg.update(_verdict_row(vd))
      if "illegal_stack" in vd["reasons"] and not bad:
        illegal += 1  # a larger cube rested on a smaller one during the move-episode
        seg["illegal_during_move"] = True
    segments.append(seg)
    if stop == "unstable" or any(h.positions()[c][2] < FELL_Z for c in H.CUBES):
      stop_reason = "unstable" if stop == "unstable" else "cube_fell_off_table"
      break
    state = seen["state"] if (seen["heights_ok"] and seen["state"] is not None) else None
  else:
    if state == H.GOAL_STATE:
      stop_reason = "solved"
  solved = state == H.GOAL_STATE
  if solved:
    stop_reason = "solved"
  h.close()
  row: dict = {"mode": mode, "hanoi": "puzzle", "run_key": str(idx), "seed": cfg["seed"],
               "attempt": idx, "episode_seed": v.seed, "instruction": H.INSTRUCTION,
               "start_state": H.START_STATE}
  all_full = all(s.get("L_full") for s in segments) if segments else False
  reasons = sorted({r for s in segments for r in s.get("reasons", [])})
  if not solved:
    reasons = ["not_solved"] + reasons
  row["rollout_file"] = _save(h, cfg, f"hanoi_puzzle_{mode}_{idx:05d}", None)
  row.update({
    "solved": solved, "moves_used": len(segments), "illegal_states": illegal,
    "solved_optimally": bool(solved and len(segments) == len(H.optimal_moves_between(H.START_STATE))),
    "final_state": state, "segments": segments,
    "pass": bool(solved and illegal == 0 and all_full), "L_task": bool(solved),
    "L_full": bool(solved and illegal == 0 and all_full), "reasons": reasons,
    "style_only": False, "stop_reason": stop_reason, "sim_s": round(float(h.sc.data.time), 3),
    "steps_10hz": h.clip["steps"], "clip": h.clip,
    "timing_s": {k: round(x, 3) for k, x in h.tm.items()}, "variation": v.to_dict()})
  if mode == "policy":
    row["inference_ms"] = {"chunks": sum(s.get("inference_chunks", 0) for s in segments),
                           "chunk_median": None}
  else:
    t0 = time.perf_counter()
    nat = run_hanoi_puzzle_expert(spec, v, max_moves=cfg["max_moves"], pinch_width=cfg["pinch_width"])
    row["expert_native"] = {"expert_verifier_pass": bool(nat["solved"]),
                            "phase_free_pass": bool(nat["solved"] and nat["illegal_states"] == 0
                                                    and all(m["L_full"] for m in nat["moves"])),
                            "moves_used": nat["moves_used"], "illegal_states": nat["illegal_states"],
                            "wall_s": round(time.perf_counter() - t0, 2)}
  row["wall_s"] = round(time.perf_counter() - t_wall, 3)
  if vid:
    row["video"] = str(vid)
  return row


def episode(item, cfg: dict, spec, G: dict) -> dict:
  idx = int(item[0] if isinstance(item, (tuple, list)) else item)
  if cfg["hanoi"] == "move":
    return move_episode(idx, cfg, spec, G)
  if cfg["hanoi"] == "puzzle":
    return puzzle_episode(idx, cfg, spec, G)
  raise ValueError(cfg["hanoi"])


def summarize(rows: list[dict], cfg: dict) -> dict:
  from collections import Counter
  n = len(rows)
  out: dict = {"task": cfg["hanoi"], "instruction": H.INSTRUCTION}
  if cfg["hanoi"] == "move":
    by_cube = {}
    for c in H.CUBES:
      sub = [r for r in rows if r["moved_cube"] == c]
      by_cube[c] = {"n": len(sub), "L_task": sum(r["L_task"] for r in sub),
                    "L_full": sum(r["L_full"] for r in sub)}
    out.update({"by_moved_cube": by_cube,
                "end_state_matches_expected": sum(r["observed_end_state"] == r["expected_end_state"]
                                                  for r in rows),
                "start_states": dict(sorted(Counter(r["start_state"] for r in rows).items()))})
  else:
    k = sum(r["solved"] for r in rows)
    out.update({"solved": k, "solved_rate": round(k / max(n, 1), 4),
                "wilson95_solved": [round(x, 4) for x in E.wilson(k, n)],
                "solved_in_7": sum(r["solved_optimally"] for r in rows),
                "moves_used": [r["moves_used"] for r in rows],
                "illegal_states_total": sum(r["illegal_states"] for r in rows),
                "runs_with_illegal_states": sum(r["illegal_states"] > 0 for r in rows),
                "max_moves": cfg["max_moves"], "move_max_s": cfg["move_max_s"]})
  return out
