"""Closed-loop scattered-stacking evaluation, run through sim_eval.py:

  # (a) one move per episode, from the datagen start buckets (the training task)
  python eval/sim_eval.py --stack move --mode policy --checkpoint CKPT/pretrained_model --n 60 --out OUT
  python eval/sim_eval.py --stack move --mode expert --n 20 --out OUT
  # (b) the whole tower in ONE continuous scene: even ids start fully scattered, odd ids
  #     from a stable wrong stack; up to --max-moves (default 8) move-episodes
  python eval/sim_eval.py --stack tower --mode policy --checkpoint CKPT/pretrained_model --n 20 --out OUT
  python eval/sim_eval.py --stack tower --mode expert --n 10 --out OUT

Stepping is hanoi_eval.HanoiHarness's (= sim_eval.run_harness_episode's): every 10 Hz step
the policy gets the measured joints (MolmoAct2 frame) and, at chunk starts, the cam_head /
cam_right renders; its action goes through the unit conversion, joint-range clip, 3-frame
"lag" expansion, 98 % servo-speed rate limit and 7-substep first-order hold. The
instruction is always stack.INSTRUCTION. Judging is the stacking verifier
(datagen/r2s/stack_verify.py) on every physics step.

(a) move. Scene = sample_stack_variation(seed, id) (the datagen draw; eval seed 1000 vs.
    training seed 1, so no overlap). A scene drawn into the "noise" bucket is run from its
    underlying start without noise (noise only perturbs the expert in datagen). A start
    whose stacks do not survive the settle is reported as start_unstable and left out of
    the rates. Policy mode stops early once a move is done (the symbolic state changed,
    every cube and the arm at rest, no jaw contact for 0.5 s, for 1 s in a row), when a
    cube falls off the table, or at --max-sim-s. Expert mode runs the scripted expert
    natively (exactly as datagen does) and replays its actions at 10 Hz through the
    harness. Each row: L_task / L_full / L_strict / reasons for the rule's next state, the
    observed end state, and the start bucket / stratum / start state.
(b) tower. Scene = sample_stack_variation(seed, id, bucket="scatter" for even ids, "wrong"
    for odd ids). Loop: read the state from the simulation; stop at the tower "TLM" or
    after --max-moves move-episodes; otherwise run one move-episode (policy: up to
    --move-max-s with the same early stop as (a); expert: plan the rule's next move on a
    COPY of the live simulation and replay it at 10 Hz in the live one). Nothing is reset
    between move-episodes. Each row: solved, moves used, advancement (fraction of the
    rule's moves from the start state that are done at the end), stop reason, failure
    reasons, and every move-episode's verdict against the rule's move from the state it
    started in (a policy may take another legal route; solved / advancement do not care).
"""

from __future__ import annotations

import copy
import dataclasses
import time
from collections import Counter
from pathlib import Path

import numpy as np

import hanoi_eval as HE
import sim_eval as E
from r2s import stack as K
from r2s.hanoi import CUBES
from r2s.hanoi_expert import TAIL_S, MoveSim, expert_move_ctx
from r2s.planner import plan_config_for
from r2s.stack_expert import (StackExpert, advancement, free_spot_now, run_stack_episode,
                              run_stack_tower_expert, stack_move_context)
from r2s.stack_verify import StackMonitor, read_stack_state, start_state_intact


class StackHarness(HE.HanoiHarness):
  """HanoiHarness with the stacking verifier and the stacking 'move done' test."""

  def __init__(self, *a, **k):
    super().__init__(*a, **k)
    self.mon = StackMonitor(self.sc, self.v)

  def new_segment(self) -> None:
    self.mon = StackMonitor(self.sc, self.v)

  def read(self) -> str | None:
    return read_stack_state(self.mon.poses())["state"]

  def move_done(self, start_state: str) -> bool:
    mon, d = self.mon, self.sc.data
    if any(mon.carrying.values()) or d.time - mon.last_jaw_contact_t <= 0.5 or not mon.at_rest():
      return False
    if float(np.max(np.abs(d.qvel[self.arm_dofs]))) > HE.ARM_STILL:
      return False
    seen = self.read()
    return seen is not None and seen != start_state


def _eval_variation(spec, cfg: dict, idx: int, **kw):
  v = K.sample_stack_variation(spec, cfg["seed"], idx, variety=True, **kw)
  info = K.stack_info(v)
  if info["bucket"] == "noise":  # eval runs the underlying start, noise-free
    info["bucket"], info["action_noise"] = info["base_bucket"], None
  return v, info


def _bucket_key(info: dict) -> str:
  return info["bucket"] + (f"/{info['stratum']}" if info.get("stratum") else "")


def _verdict_row(vd: dict) -> dict:
  return {"L_task": vd["L_task"], "L_full": vd["L_full"], "L_strict": vd["L_strict"],
          "reasons": vd["reasons"], "style_only": vd["style_only"],
          "observed_end_state": vd["observed_end_state"],
          "expected_end_state": vd["expected_end_state"], "moved_cube": vd["moved_cube"]}


def _video_path(cfg: dict, idx: int, kind: str) -> Path | None:
  if str(idx) not in cfg.get("video_ids", []):
    return None
  vdir = Path(cfg["out"]) / "videos"
  vdir.mkdir(parents=True, exist_ok=True)
  return vdir / f"stack_{kind}_{cfg['mode']}_{idx:05d}.mp4"


def _unstable_start_row(row: dict, h: StackHarness, intact: dict, v, t_wall: float) -> dict:
  h.close()
  row.update({"valid_scene": False, "pass": False, "L_task": False, "L_full": False,
              "L_strict": False, "reasons": ["start_unstable"], "style_only": False,
              "stop_reason": "start_unstable", "start_check": intact["seen"],
              "sim_s": round(float(h.sc.data.time), 3), "steps_10hz": 0, "clip": h.clip,
              "timing_s": {k: round(x, 3) for k, x in h.tm.items()},
              "inference_ms": {"chunks": 0, "chunk_median": None, "chunk_max": None},
              "expert_native": {"expert_verifier_pass": False, "phase_free_pass": False},
              "variation": v.to_dict(), "wall_s": round(time.perf_counter() - t_wall, 3)})
  return row


def _expert_actions_on_copy(h: StackHarness, spec, state: str, move, pinch_width,
                            rng: np.random.Generator) -> tuple[np.ndarray, list, dict]:
  """Plan and run the scripted expert's move on a COPY of the live simulation (the live
  scene is not touched); return its 10 Hz actions in the MolmoAct2 frame."""
  data = copy.copy(h.sc.data)
  sc2 = dataclasses.replace(h.sc, data=data)
  sim = MoveSim(sc2, h.v, monitor_cls=StackMonitor)
  spot = free_spot_now(rng, sc2, sim.ids, state, move[0]) if move[1] == K.TABLE else None
  ctx = stack_move_context(sc2, sim.ids, state, tuple(move), spot)
  vm = K.variation_for_move(h.v, state, tuple(move), spot)
  info, fails = expert_move_ctx(sim, spec, h.v, ctx, vm, plan_config_for(h.v), pinch_width,
                                start_target=h.last_frame, expert_cls=StackExpert)
  if sim.rec.action and not sim.unstable():
    sim.hold(TAIL_S)
  a30 = np.array(sim.rec.action, float).reshape(-1, 6)
  return E.sim_to_mm2(a30[::E.STEP_FRAMES], h.cal), fails, info


def move_episode(idx: int, cfg: dict, spec, G: dict) -> dict:
  t_wall = time.perf_counter()
  mode = cfg["mode"]
  if mode not in ("policy", "expert"):
    raise ValueError(f"--stack move supports --mode policy|expert, not {mode}")
  v, info = _eval_variation(spec, cfg, idx)
  vid = _video_path(cfg, idx, "move")
  row: dict = {"mode": mode, "stack": "move", "run_key": str(idx), "seed": cfg["seed"],
               "attempt": idx, "episode_seed": v.seed, "instruction": K.INSTRUCTION,
               "bucket": info["bucket"], "stratum": info["stratum"], "bucket_key": _bucket_key(info),
               "start_state": info["start_state"], "move": info["move"]}
  h = StackHarness(spec, v, quality=cfg["quality"], obs_codec=cfg["obs_codec"],
                   need_images=(mode == "policy"), video=vid)
  intact = start_state_intact(h.mon.poses(), info["start_state"])
  if not intact["ok"]:
    return _unstable_start_row(row, h, intact, v, t_wall)
  row["valid_scene"] = True
  if mode == "policy":
    src, run = HE._source(cfg, G)
    stop, _ = h.run_segment(src, cfg["max_sim_s"], info["start_state"], early_stop=True,
                            interp=cfg["interp"])
    row["inference_ms"] = {"chunks": len(run.chunk_ms),
                           "chunk_median": round(float(np.median(run.chunk_ms)), 1) if run.chunk_ms else None,
                           "chunk_max": round(float(np.max(run.chunk_ms)), 1) if run.chunk_ms else None}
    run.chunk_ms.clear()
    run.call_ms.clear()
  else:
    t0 = time.perf_counter()
    res = run_stack_episode(spec, v, pinch_width=cfg["pinch_width"])
    a10 = (E.sim_to_mm2(res.action[::E.STEP_FRAMES]) if res.action is not None
           else np.zeros((0, 6), np.float32))
    st = res.stats.get("stack", {})
    row["expert_native"] = {"expert_verifier_pass": bool(res.accepted), "expert_reasons": res.reasons,
                            "phase_free_pass": bool(st.get("L_full")),
                            "L_strict": bool(st.get("L_strict")),
                            "frames_30hz": 0 if res.action is None else int(len(res.action)),
                            "tilt_deg": res.plan_info.get("tilt_choice_deg"),
                            "wall_s": round(time.perf_counter() - t0, 2)}
    stop, _ = h.run_segment(E.ReplaySource(a10), 1e6, info["start_state"], early_stop=False,
                            interp=cfg["interp"], tail_s=cfg["tail_s"])
  vd = h.mon.verdict(info["end_state"], info["moved_cube"], info["move"][1])
  h.close()
  row["rollout_file"] = HE._save(h, cfg, f"stack_move_{mode}_{idx:05d}", info["moved_cube"])
  row.update({"pass": vd["passed"], **_verdict_row(vd), "stop_reason": stop,
              "sim_s": round(float(h.sc.data.time), 3), "steps_10hz": h.clip["steps"],
              "clip": h.clip, "timing_s": {k: round(x, 3) for k, x in h.tm.items()},
              "verify_stats": vd["stats"], "variation": v.to_dict(),
              "wall_s": round(time.perf_counter() - t_wall, 3)})
  if vid:
    row["video"] = str(vid)
  return row


def tower_episode(idx: int, cfg: dict, spec, G: dict) -> dict:
  t_wall = time.perf_counter()
  mode = cfg["mode"]
  if mode not in ("policy", "expert"):
    raise ValueError(f"--stack tower supports --mode policy|expert, not {mode}")
  start_kind = "scatter" if idx % 2 == 0 else "wrong"
  v, info = _eval_variation(spec, cfg, idx, bucket=start_kind)
  start = info["start_state"]
  vid = _video_path(cfg, idx, "tower")
  row: dict = {"mode": mode, "stack": "tower", "run_key": str(idx), "seed": cfg["seed"],
               "attempt": idx, "episode_seed": v.seed, "instruction": K.INSTRUCTION,
               "start_kind": start_kind, "stratum": info["stratum"], "start_state": start,
               "rule_moves": len(K.rule_moves_to_goal(start))}
  h = StackHarness(spec, v, quality=cfg["quality"], obs_codec=cfg["obs_codec"],
                   need_images=(mode == "policy"), video=vid)
  intact = start_state_intact(h.mon.poses(), start)
  if not intact["ok"]:
    row.update({"solved": False, "moves_used": 0, "advancement": 0.0, "final_state": None,
                "segments": []})
    return _unstable_start_row(row, h, intact, v, t_wall)
  rng = np.random.default_rng([v.seed, 23])
  state, segments, stop_reason = start, [], "max_moves"
  for k in range(cfg["max_moves"]):
    if state == K.GOAL_STATE:
      break
    if state is None:
      stop_reason = "unreadable_state"
      break
    move = K.next_move(state)
    expected = K.apply_move(state, move)
    h.new_segment()
    seg: dict = {"k": k, "from_state": state, "expected_move": list(move),
                 "expected_state": expected}
    if mode == "policy":
      src, run = HE._source(cfg, G)
      stop, seg_s = h.run_segment(src, cfg["move_max_s"], state, early_stop=True,
                                  interp=cfg["interp"])
      seg["inference_chunks"] = len(run.chunk_ms)
      run.chunk_ms.clear()
      run.call_ms.clear()
    else:
      a10, fails, pinfo = _expert_actions_on_copy(h, spec, state, move, cfg["pinch_width"], rng)
      seg["expert_planner_failures"] = fails
      seg["tilt_deg"] = pinfo.get("tilt_choice_deg")
      stop, seg_s = h.run_segment(E.ReplaySource(a10), 1e6, state, early_stop=False,
                                  interp=cfg["interp"], tail_s=cfg["tail_s"])
    vd = h.mon.verdict(expected, move[0], move[1])
    seg.update({"stop_reason": stop, "seg_sim_s": seg_s, **_verdict_row(vd)})
    segments.append(seg)
    state = vd["observed_end_state"]
    if stop == "unstable" or any(h.positions()[c][2] < HE.FELL_Z for c in CUBES):
      stop_reason = "unstable" if stop == "unstable" else "cube_fell_off_table"
      break
  solved = state == K.GOAL_STATE
  if solved:
    stop_reason = "solved"
  h.close()
  reasons = sorted({r for s in segments for r in s.get("reasons", [])})
  if not solved:
    reasons = ["not_solved", stop_reason] + [r for r in reasons if r != stop_reason]
  all_full = bool(segments) and all(s["L_full"] for s in segments)
  row["rollout_file"] = HE._save(h, cfg, f"stack_tower_{mode}_{idx:05d}", None)
  row.update({
    "valid_scene": True, "solved": solved, "moves_used": len(segments),
    "advancement": round(advancement(start, state), 4), "final_state": state,
    "segments": segments, "pass": bool(solved and all_full), "L_task": bool(solved),
    "L_full": bool(solved and all_full), "L_strict": bool(solved and all(s["L_strict"] for s in segments)),
    "reasons": reasons, "style_only": False, "stop_reason": stop_reason,
    "sim_s": round(float(h.sc.data.time), 3), "steps_10hz": h.clip["steps"], "clip": h.clip,
    "timing_s": {k: round(x, 3) for k, x in h.tm.items()}, "variation": v.to_dict()})
  if mode == "policy":
    row["inference_ms"] = {"chunks": sum(s.get("inference_chunks", 0) for s in segments),
                           "chunk_median": None}
  else:
    t0 = time.perf_counter()
    nat = run_stack_tower_expert(spec, v, max_moves=cfg["max_moves"], pinch_width=cfg["pinch_width"])
    row["expert_native"] = {"expert_verifier_pass": bool(nat["solved"]),
                            "phase_free_pass": bool(nat["solved"] and all(m["L_full"] for m in nat["moves"])),
                            "moves_used": nat["moves_used"], "stop_reason": nat["stop_reason"],
                            "advancement": nat["advancement"],
                            "wall_s": round(time.perf_counter() - t0, 2)}
  row["wall_s"] = round(time.perf_counter() - t_wall, 3)
  if vid:
    row["video"] = str(vid)
  return row


def episode(item, cfg: dict, spec, G: dict) -> dict:
  idx = int(item[0] if isinstance(item, (tuple, list)) else item)
  if cfg["stack"] == "move":
    return move_episode(idx, cfg, spec, G)
  if cfg["stack"] == "tower":
    return tower_episode(idx, cfg, spec, G)
  raise ValueError(cfg["stack"])


def _rates(sub: list[dict], keys=("L_task", "L_full", "L_strict")) -> dict:
  n = len(sub)
  out = {"n": n}
  for k in keys:
    c = sum(bool(r.get(k)) for r in sub)
    out[k] = c
    out[f"{k}_rate"] = round(c / max(n, 1), 4)
  return out


def summarize(rows: list[dict], cfg: dict) -> dict:
  valid = [r for r in rows if r.get("valid_scene", True)]
  out: dict = {"task": cfg["stack"], "instruction": K.INSTRUCTION,
               "scenes": len(rows), "valid_scenes": len(valid),
               "start_unstable": len(rows) - len(valid), "all_valid": _rates(valid)}
  if cfg["stack"] == "move":
    out["by_bucket"] = {b: _rates([r for r in valid if r["bucket"] == b])
                        for b in sorted({r["bucket"] for r in valid})}
    out["by_bucket_stratum"] = {b: _rates([r for r in valid if r["bucket_key"] == b])
                                for b in sorted({r["bucket_key"] for r in valid})}
    out["by_start_state"] = {s: _rates([r for r in valid if r["start_state"] == s])
                             for s in sorted({r["start_state"] for r in valid})}
    out["end_state_matches_expected"] = sum(r["observed_end_state"] == r["expected_end_state"]
                                            for r in valid)
    out["reason_counts"] = dict(Counter(x for r in valid for x in r["reasons"]).most_common())
  else:
    def tower(sub):
      k = sum(r["solved"] for r in sub)
      return {"n": len(sub), "solved": k, "solved_rate": round(k / max(len(sub), 1), 4),
              "wilson95_solved": [round(x, 4) for x in E.wilson(k, len(sub))],
              "solved_in_rule_moves": sum(r["solved"] and r["moves_used"] == r["rule_moves"] for r in sub),
              "moves_used": [r["moves_used"] for r in sub],
              "moves_used_mean_solved": round(float(np.mean([r["moves_used"] for r in sub if r["solved"]])), 2)
              if k else None,
              "advancement_mean": round(float(np.mean([r["advancement"] for r in sub])), 4) if sub else None,
              "stop_reasons": dict(Counter(r["stop_reason"] for r in sub)),
              "failure_reasons": dict(Counter(x for r in sub if not r["solved"]
                                              for x in r["reasons"]).most_common())}
    out["all"] = tower(valid)
    out["by_start_kind"] = {s: tower([r for r in valid if r["start_kind"] == s])
                            for s in sorted({r["start_kind"] for r in valid})}
    out["max_moves"], out["move_max_s"] = cfg["max_moves"], cfg["move_max_s"]
  return out
