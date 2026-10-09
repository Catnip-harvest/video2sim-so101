"""Check that a converted object behaves physically before any demo is generated from it.

Three tests, all CPU MuJoCo with the datagen physics (timestep 1/210 s, implicitfast,
elliptic cones, impratio 10), no rendering:

  settle   Drop the object 5 mm onto the table and wait 2 s. Pass marks (fixed in advance):
           resting height within 2 mm of its half height (no sinking or floating), contact
           penetration <= 1 mm, in the last 0.5 s speed < 5 mm/s and spin < 3 deg/s (no
           jitter), drift over the last 1 s < 1 mm, tilt from upright < 3 deg.
  parity   Run the existing physics-verified scripted generator (datagen/r2s/rollout.py, the
           same verifier that gates every dataset episode) on the SAME seeds with object A and
           object B. Each seed fixes the object pose, servo strengths, frictions and mass
           factor, so the two runs differ only in the object. Reports, per object, with 95 %
           Wilson intervals: grasp+lift success (both jaws on it and >= 1 cm up after the
           lift), carried without a drop, full episode accepted; plus in-hand slip and lift
           height, and the paired disagreement counts.
  scale    The same batch at scale x0.95, 0.97, 1.03, 1.05 (mass unchanged: the kitchen scale
           is right even when the size estimate is not). Shows which scale error breaks
           grasping first.

Objects are given as either a folder/object.xml written by mesh_to_mjcf.py, or
``box:EDGE_M[:MASS_KG]`` for the primitive cube the datagen uses today.

    python -I object3d/validate_object.py settle object3d/out/ycb_077_rubiks_cube
    python -I object3d/validate_object.py parity --a box:0.057:0.094 --b object3d/out/ycb_077_rubiks_cube --n 30
    python -I object3d/validate_object.py scale object3d/out/ycb_077_rubiks_cube --n 30
    python -I object3d/validate_object.py all object3d/out/ycb_077_rubiks_cube --n 30 --out object3d/results/ycb_077
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATAGEN = REPO / "datagen"
DEFAULT_SPEC = REPO / "demo" / "outputs" / "pexels_7488188_sugar_cube" / "so101_task_spec.json"


def _datagen_path() -> None:
  if str(DATAGEN) not in sys.path:
    sys.path.insert(0, str(DATAGEN))


# --------------------------------------------------------------------------------------
# objects
# --------------------------------------------------------------------------------------
@dataclass
class ObjectSource:
  """Either a primitive box (edge, mass) or a mesh object folder; plus a scale factor."""
  text: str
  scale: float = 1.0

  @property
  def is_box(self) -> bool:
    return self.text.startswith("box:")

  def box(self) -> tuple[float, float]:
    parts = self.text.split(":")
    edge = float(parts[1]) * self.scale
    mass = float(parts[2]) if len(parts) > 2 else 0.025
    return edge, mass

  def mesh(self):
    _datagen_path()
    from r2s.mesh_object import MeshObject

    return MeshObject.from_path(self.text, scale_mult=self.scale)

  def label(self) -> str:
    base = self.text if self.is_box else Path(self.text).name
    return base if abs(self.scale - 1) < 1e-9 else f"{base} x{self.scale:g}"

  def half_extents(self) -> np.ndarray:
    if self.is_box:
      e = self.box()[0]
      return np.array([e / 2] * 3)
    return self.mesh().half

  def mass(self) -> float:
    return self.box()[1] if self.is_box else self.mesh().mass


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
  """95 % Wilson score interval for k successes out of n."""
  if n == 0:
    return (0.0, 1.0)
  p = k / n
  den = 1 + z * z / n
  c = (p + z * z / (2 * n)) / den
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------------------------------
# (a) settle
# --------------------------------------------------------------------------------------
SETTLE_PASS = {"height_err_mm": 2.0, "penetration_mm": 1.0, "speed_mm_s": 5.0,
               "spin_deg_s": 3.0, "drift_mm": 1.0, "tilt_deg": 3.0}


def settle_test(src: ObjectSource, seconds: float = 2.0, drop: float = 0.005, yaw: float = 0.3,
                friction: float = 0.7) -> dict:
  _datagen_path()
  import mujoco
  from r2s import scene as S
  from r2s.paths import mj_scene

  spec = mujoco.MjSpec()
  spec.option.timestep = S.TIMESTEP
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = mj_scene.IMPRATIO
  spec.option.iterations = mj_scene.SOLVER_ITERATIONS
  spec.option.ls_iterations = mj_scene.SOLVER_LS_ITERATIONS
  # Same table and contact settings as a datagen episode with this object
  # (r2s/scene.py: mesh objects get multiccd and a box table top).
  S.add_table(spec, mesh_object=not src.is_box)
  if not src.is_box:
    S.enable_mesh_contacts(spec)
  half = src.half_extents()
  if src.is_box:
    edge, mass = src.box()
    obj = mj_scene.object_spec(kind="box", size=edge, mass=mass, friction=(friction, 0.005, 0.0001))
  else:
    m = src.mesh()
    obj = m.object_spec(mass=m.mass, friction=(friction, 0.005, 0.0001), with_visual=False)
  obj.body("object").pos = (0.2, 0.0, float(half[2]) + drop)
  obj.body("object").quat = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
  spec.attach(obj, prefix="", frame=spec.worldbody.add_frame())
  model = spec.compile()
  data = mujoco.MjData(model)
  b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
  j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
  dadr = model.jnt_dofadr[j]
  mujoco.mj_forward(model, data)
  n = int(round(seconds / model.opt.timestep))
  pos = np.zeros((n, 3))
  zax = np.zeros((n, 3))
  lin = np.zeros(n)
  ang = np.zeros(n)
  pen = np.zeros(n)
  t0 = time.perf_counter()
  for k in range(n):
    mujoco.mj_step(model, data)
    pos[k] = data.xpos[b]
    zax[k] = data.xmat[b].reshape(3, 3)[:, 2]
    lin[k] = np.linalg.norm(data.qvel[dadr:dadr + 3])
    ang[k] = np.linalg.norm(data.qvel[dadr + 3:dadr + 6])
    pen[k] = max(0.0, -float(data.contact.dist[:data.ncon].min())) if data.ncon else 0.0
  wall = time.perf_counter() - t0
  last_half = slice(n - int(0.5 / model.opt.timestep), n)
  last_1s = int(1.0 / model.opt.timestep)
  still = lin < 0.005
  settled_at = None
  for k in range(n):
    if still[k:].all():
      settled_at = k * model.opt.timestep
      break
  res = {
    "object": src.label(),
    "half_height_mm": round(float(half[2]) * 1e3, 2),
    "height_err_mm": round((pos[-1, 2] - half[2]) * 1e3, 3),
    "penetration_mm": round(float(pen[-1]) * 1e3, 3),
    "max_penetration_mm": round(float(pen.max()) * 1e3, 3),
    "speed_mm_s": round(float(lin[last_half].max()) * 1e3, 3),
    "spin_deg_s": round(float(math.degrees(ang[last_half].max())), 3),
    "drift_mm": round(float(np.linalg.norm(pos[-1] - pos[-last_1s])) * 1e3, 3),
    # context, not a pass mark: how far the body actually moves while it "jitters"
    "jitter_peak_to_peak_um": round(float(np.ptp(pos[last_half], axis=0).max()) * 1e6, 2),
    "tilt_deg": round(float(math.degrees(math.acos(np.clip(zax[-1, 2], -1, 1)))), 3),
    "settled_after_s": None if settled_at is None else round(settled_at, 3),
    "n_contacts_at_rest": int(data.ncon),
    "seconds": seconds, "drop_mm": drop * 1e3, "wall_s": round(wall, 3),
    "sim_speed_x_realtime": round(seconds / wall, 1),
  }
  fails = [k for k, lim in SETTLE_PASS.items() if abs(res[k]) > lim]
  res["pass_marks"] = SETTLE_PASS
  res["failed"] = fails
  res["passed"] = not fails
  return res


# --------------------------------------------------------------------------------------
# (b) grasp batch on the datagen generator
# --------------------------------------------------------------------------------------
_W: dict = {}


def _build_spec(spec_path: str, src: ObjectSource):
  _datagen_path()
  from r2s.mesh_object import load_spec, spec_with_mesh

  if src.is_box:
    spec = load_spec(spec_path)
    edge, mass = src.box()
    spec.object.size = edge
    spec.object.height = edge
    spec.object.mass = mass
    return spec
  d = json.loads(Path(spec_path).read_text(encoding="utf-8"))
  return spec_with_mesh(d, src.mesh(), spec_path)


def _init(cfg: dict) -> None:
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  _datagen_path()
  _W["cfg"] = cfg
  _W["specs"] = {}


def _one(job: tuple[str, float, int]) -> dict:
  import gc

  try:
    return _one_inner(job)
  finally:
    # An episode leaves its MjModel/MjData copies in reference cycles (the planner's
    # closures). With mesh objects that is ~250 MB per episode until the cyclic GC runs;
    # six workers then exhausted the commit limit and qhull's malloc failed inside MuJoCo's
    # compile (measured 9 Oct 2026). Collect after every episode.
    gc.collect()


def _one_inner(job: tuple[str, float, int]) -> dict:
  text, scale, attempt = job
  _datagen_path()
  from r2s.planner import PlanConfig
  from r2s.rollout import run_episode
  from r2s.variations import sample_variation

  cfg = _W["cfg"]
  key = (text, scale)
  if key not in _W["specs"]:
    _W["specs"][key] = _build_spec(cfg["spec"], ObjectSource(text, scale))
  spec = _W["specs"][key]
  src = ObjectSource(text, scale)
  v = sample_variation(spec, cfg["seed"], attempt)
  if src.is_box:  # the variation clamps cube size to 20-50 mm; parity needs the exact edge
    v.obj_size = src.box()[0]
  pc = PlanConfig(**cfg.get("plan", {}))
  t0 = time.perf_counter()
  try:
    r = run_episode(spec, v, cfg=pc, pinch_width=cfg.get("pinch_width"), keep_arrays=False)
  except Exception as e:  # noqa: BLE001  (counted as a failure with its message)
    return {"object": src.label(), "text": text, "scale": scale, "attempt": attempt,
            "accepted": False, "primary_reason": "exception", "reasons": ["exception"],
            "error": f"{type(e).__name__}: {e}", "stats": {}, "wall_s": time.perf_counter() - t0}
  st = r.stats
  lift = st.get("lift_check_mm")
  grasp_ok = lift is not None and lift >= 10.0 and "grasp_failed" not in r.reasons
  return {
    "object": src.label(), "text": text, "scale": scale, "attempt": attempt,
    "accepted": bool(r.accepted), "primary_reason": r.primary_reason, "reasons": r.reasons,
    "grasp_ok": bool(grasp_ok),
    "carried_ok": bool(grasp_ok and "dropped_or_slipped" not in r.reasons),
    "lift_check_mm": lift, "max_obj_lift_mm": st.get("max_obj_lift_mm"),
    "in_hand_slip_mm": st.get("in_hand_slip_mm"),
    "squeezed_width_mm": (r.plan_info.get("pick") or {}).get("squeezed_width_mm"),
    "jaw_after_squeeze_rad": st.get("jaw_angle_after_squeeze"),
    "plan_error": r.plan_info.get("error"),
    "servo": v.servo_variant, "sim_s": round(r.sim_seconds, 2),
    "wall_s": round(time.perf_counter() - t0, 2),
  }


def run_batch(jobs: list[tuple[str, float, int]], spec_path: str, seed: int, workers: int,
              plan: dict | None = None) -> list[dict]:
  _datagen_path()
  cfg = {"spec": str(spec_path), "seed": seed, "plan": plan or {}}
  try:
    from generate import pinch_width_from_spec  # same open width rule as the generator

    cfg["pinch_width"] = pinch_width_from_spec(Path(spec_path))
  except Exception:  # noqa: BLE001
    cfg["pinch_width"] = None
  if workers <= 1:
    _init(cfg)
    return [_one(j) for j in jobs]
  import multiprocessing as mp

  out = []
  t0 = time.perf_counter()
  # maxtasksperchild: fresh workers every 10 episodes, so memory stays bounded
  with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(cfg,),
                                    maxtasksperchild=10) as pool:
    for r in pool.imap(_one, jobs, chunksize=1):
      out.append(r)
      if len(out) % 10 == 0 or len(out) == len(jobs):
        print(f"[batch] {len(out)}/{len(jobs)} episodes, {time.perf_counter() - t0:.0f} s",
              file=sys.stderr, flush=True)
  return out


def summarise(rows: list[dict]) -> dict:
  n = len(rows)
  out = {"n": n}
  for key in ("grasp_ok", "carried_ok", "accepted"):
    k = sum(bool(r.get(key)) for r in rows)
    lo, hi = wilson(k, n)
    out[key] = {"k": k, "rate": round(k / n, 3) if n else None,
                "wilson95": [round(lo, 3), round(hi, 3)]}
  slip = [r["in_hand_slip_mm"] for r in rows if r.get("in_hand_slip_mm") is not None]
  lift = [r["max_obj_lift_mm"] for r in rows if r.get("grasp_ok") and r.get("max_obj_lift_mm") is not None]
  out["in_hand_slip_mm"] = _q(slip)
  out["max_lift_mm_when_grasped"] = _q(lift)
  reasons: dict[str, int] = {}
  for r in rows:
    if not r["accepted"]:
      reasons[r["primary_reason"] or "?"] = reasons.get(r["primary_reason"] or "?", 0) + 1
  out["rejections_primary"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
  out["wall_s_per_episode_median"] = _q([r["wall_s"] for r in rows])["median"]
  return out


def _q(x: list[float]) -> dict:
  if not x:
    return {"n": 0, "median": None, "p90": None, "max": None}
  a = np.asarray(x, float)
  return {"n": int(len(a)), "median": round(float(np.median(a)), 2),
          "p90": round(float(np.quantile(a, 0.9)), 2), "max": round(float(a.max()), 2)}


def paired(rows_a: list[dict], rows_b: list[dict], key: str) -> dict:
  a = {r["attempt"]: bool(r.get(key)) for r in rows_a}
  b = {r["attempt"]: bool(r.get(key)) for r in rows_b}
  common = sorted(set(a) & set(b))
  both = sum(a[i] and b[i] for i in common)
  only_a = sum(a[i] and not b[i] for i in common)
  only_b = sum(b[i] and not a[i] for i in common)
  n = len(common)
  diff = (only_a - only_b) / n if n else 0.0
  return {"n": n, "both": both, "only_a": only_a, "only_b": only_b,
          "neither": n - both - only_a - only_b, "rate_a_minus_b": round(diff, 3)}


def parity(a: ObjectSource, b: ObjectSource, n: int, spec: str, seed: int, workers: int,
           plan: dict | None = None) -> dict:
  jobs = [(a.text, a.scale, i) for i in range(n)] + [(b.text, b.scale, i) for i in range(n)]
  t0 = time.perf_counter()
  rows = run_batch(jobs, spec, seed, workers, plan)
  ra = [r for r in rows if r["text"] == a.text and r["scale"] == a.scale]
  rb = [r for r in rows if r["text"] == b.text and r["scale"] == b.scale]
  return {"a": a.label(), "b": b.label(), "seed": seed, "spec": str(spec), "plan": plan or {},
          "summary_a": summarise(ra), "summary_b": summarise(rb),
          "paired": {k: paired(ra, rb, k) for k in ("grasp_ok", "carried_ok", "accepted")},
          "wall_s": round(time.perf_counter() - t0, 1), "rows": rows}


def scale_sweep(src: ObjectSource, scales: list[float], n: int, spec: str, seed: int,
                workers: int, plan: dict | None = None) -> dict:
  all_scales = [1.0] + [s for s in scales if abs(s - 1.0) > 1e-9]
  jobs = [(src.text, s, i) for s in all_scales for i in range(n)]
  t0 = time.perf_counter()
  rows = run_batch(jobs, spec, seed, workers, plan)
  by = {}
  base = [r for r in rows if r["scale"] == 1.0]
  for s in all_scales:
    rs = [r for r in rows if r["scale"] == s]
    by[f"{s:g}"] = {"summary": summarise(rs),
                    "paired_vs_1": {k: paired(base, rs, k) for k in ("grasp_ok", "accepted")}}
  return {"object": src.label(), "seed": seed, "spec": str(spec), "plan": plan or {},
          "by_scale": by, "wall_s": round(time.perf_counter() - t0, 1), "rows": rows}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _fmt_rate(s: dict, key: str) -> str:
  d = s[key]
  return f"{d['k']}/{s['n']} ({100 * d['rate']:.0f} %, CI {100 * d['wilson95'][0]:.0f}-{100 * d['wilson95'][1]:.0f})"


def markdown(rep: dict) -> str:
  lines = []
  if "settle" in rep:
    s = rep["settle"]
    lines += [f"### Settle: {'PASS' if s['passed'] else 'FAIL ' + ', '.join(s['failed'])}", "",
              "| height err mm | penetration mm | speed mm/s (last 0.5 s) | spin deg/s | drift mm (last 1 s) | tilt deg | jitter p-p um | settled after s |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|",
              f"| {s['height_err_mm']} | {s['penetration_mm']} | {s['speed_mm_s']} | {s['spin_deg_s']} | {s['drift_mm']} | {s['tilt_deg']} | {s['jitter_peak_to_peak_um']} | {s['settled_after_s']} |", ""]
  if "parity" in rep:
    p = rep["parity"]
    lines += [f"### Parity: A = {p['a']}, B = {p['b']} (seed {p['seed']}, same draws)", "",
              "| | grasp + lift | carried, no drop | full episode accepted | slip mm median/p90 | lift mm median | top rejections |",
              "|---|---|---|---|---|---|---|"]
    for tag in ("a", "b"):
      s = p[f"summary_{tag}"]
      lines.append(f"| {p[tag]} | {_fmt_rate(s, 'grasp_ok')} | {_fmt_rate(s, 'carried_ok')} | "
                   f"{_fmt_rate(s, 'accepted')} | {s['in_hand_slip_mm']['median']}/{s['in_hand_slip_mm']['p90']} | "
                   f"{s['max_lift_mm_when_grasped']['median']} | {s['rejections_primary']} |")
    pa = p["paired"]["accepted"]
    lines += ["", f"Paired (accepted): both {pa['both']}, only A {pa['only_a']}, only B {pa['only_b']}, "
              f"neither {pa['neither']}; A - B = {100 * pa['rate_a_minus_b']:.0f} points.", ""]
  if "scale" in rep:
    sc = rep["scale"]
    lines += [f"### Scale sensitivity: {sc['object']}", "",
              "| scale | grasp + lift | full episode accepted | slip mm median | top rejections |",
              "|---|---|---|---|---|"]
    for k, d in sc["by_scale"].items():
      s = d["summary"]
      lines.append(f"| x{k} | {_fmt_rate(s, 'grasp_ok')} | {_fmt_rate(s, 'accepted')} | "
                   f"{s['in_hand_slip_mm']['median']} | {s['rejections_primary']} |")
    lines.append("")
  return "\n".join(lines)


def main(argv=None) -> None:
  ap = argparse.ArgumentParser(description="settle / grasp-parity / scale tests for an object")
  sub = ap.add_subparsers(dest="cmd", required=True)

  def common(p, batch=True):
    p.add_argument("--out", help="folder for report.json + report.md")
    if batch:
      p.add_argument("--n", type=int, default=30, help="seeds (episodes) per object/scale")
      p.add_argument("--seed", type=int, default=0)
      p.add_argument("--workers", type=int, default=4)
      p.add_argument("--spec", default=str(DEFAULT_SPEC))
      p.add_argument("--plan", default="{}", help='PlanConfig overrides as JSON, e.g. '
                     '\'{"open_max_rad": 1.6, "grasp_height_max": 0.03}\'')

  p = sub.add_parser("settle")
  p.add_argument("obj")
  p.add_argument("--seconds", type=float, default=2.0)
  common(p, batch=False)
  p = sub.add_parser("parity")
  p.add_argument("--a", required=True)
  p.add_argument("--b", required=True)
  common(p)
  p = sub.add_parser("scale")
  p.add_argument("obj")
  p.add_argument("--scales", type=float, nargs="+", default=[0.95, 0.97, 1.03, 1.05])
  common(p)
  p = sub.add_parser("all", help="settle + grasp batch (+ parity vs --ref) + scale sweep")
  p.add_argument("obj")
  p.add_argument("--ref", help="reference object for parity (e.g. box:0.057:0.094)")
  p.add_argument("--scales", type=float, nargs="+", default=[0.95, 0.97, 1.03, 1.05])
  p.add_argument("--no-scale", action="store_true")
  common(p)
  a = ap.parse_args(argv)

  rep: dict = {"cmd": a.cmd, "argv": sys.argv[1:]}
  if a.cmd in ("settle", "all"):
    rep["settle"] = settle_test(ObjectSource(a.obj), seconds=getattr(a, "seconds", 2.0))
  plan = json.loads(getattr(a, "plan", "{}") or "{}")
  if a.cmd == "parity":
    rep["parity"] = parity(ObjectSource(a.a), ObjectSource(a.b), a.n, a.spec, a.seed, a.workers, plan)
  if a.cmd == "all" and a.ref:
    rep["parity"] = parity(ObjectSource(a.ref), ObjectSource(a.obj), a.n, a.spec, a.seed,
                           a.workers, plan)
  if a.cmd == "scale" or (a.cmd == "all" and not a.no_scale):
    rep["scale"] = scale_sweep(ObjectSource(a.obj), a.scales, a.n, a.spec, a.seed, a.workers, plan)
  md = markdown(rep)
  print(md)
  if a.out:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
  main()
