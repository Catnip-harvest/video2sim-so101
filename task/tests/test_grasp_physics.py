"""Grasp-physics honesty test for the SO-101 pick-and-place task (CPU MuJoCo only).

Why this exists: an earlier project in this repo "grasped" with a kinematic latch that
rewrote the object's pose every step; the jaws never touched the object and nobody
noticed. This test proves the opposite for the model the RL task trains on:

* the model contains no equality constraints (weld/connect), the object is a free body
  (not mocap) and nothing ever writes the object's state after the initial placement;
* a scripted IK pick (position servos only, joint targets interpolated) lifts the
  object, and while it is airborne it touches **both** jaws and nothing else, and the
  contact forces on it balance its weight;
* when the gripper opens, the object falls back to the table;
* a negative control (same script, gripper never closes) never lifts the object;
* the grasp still works at the corners of the domain-randomisation range
  (weak 7.4 V servo, low finger friction, heavy / small / large object).

Run:
  python -m pytest -q manip/task/tests/test_grasp_physics.py         (needs pytest)
  python -I manip/task/tests/test_grasp_physics.py                    (writes JSON)
Only mujoco + numpy are needed (no torch / warp / mjlab).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
  sys.path.insert(0, str(TASK_ROOT))

from so101_pickplace import ik, mj_scene  # noqa: E402
from so101_pickplace import robot_model as rm  # noqa: E402

RESULT_JSON = TASK_ROOT / "measurements" / "grasp_physics_result.json"
FIXED_JAW = re.compile(r"fixed_jaw_.*")
MOVING_JAW = re.compile(r"moving_jaw_.*")
LIFT_HEIGHT = 0.08  # grasp-site rise after closing (m)
PAN_AXIS_X = 0.0388


@dataclass
class PickCase:
  name: str
  radius: float = 0.20  # from the pan axis
  lateral: float = 0.03
  object_yaw: float = 0.3
  object_kind: str = "box"
  object_size: float = 0.030
  object_height: float = 0.040  # cylinder only
  object_mass: float = 0.030
  finger_friction: float = 1.0
  gripper_effort: float = rm.GRIPPER_EFFORT_LIMIT
  arm_effort: float = rm.ARM_EFFORT_LIMIT
  close_gripper: bool = True


@dataclass
class PickResult:
  case: str
  ik_pos_err_mm: float
  object_start_z: float
  object_max_z: float
  object_hold_z: float
  lifted_by_m: float
  airborne_steps: int
  airborne_steps_with_both_jaws: int
  airborne_steps_with_foreign_contact: int
  hold_normal_force_fixed_jaw_n: float
  hold_normal_force_moving_jaw_n: float
  hold_net_contact_force_z_n: float
  object_weight_n: float
  slip_during_hold_mm: float
  gripper_angle_closed_rad: float
  object_z_after_open: float
  fell_within_s: float | None
  finger_contacts_after_open: int
  model_neq: int
  object_is_mocap: bool
  notes: list[str] = field(default_factory=list)


def _face_normal_closest(yaw: float, current: np.ndarray) -> np.ndarray:
  best, best_dot = None, -2.0
  for k in range(4):
    a = yaw + k * np.pi / 2
    n = np.array([np.cos(a), np.sin(a), 0.0])
    if np.dot(n, current) > best_dot:
      best, best_dot = n, np.dot(n, current)
  return best


class _Rig:
  def __init__(self, case: PickCase):
    self.case = case
    half = mj_scene.object_half_height(case.object_kind, case.object_size, case.object_height)
    self.obj_start = np.array([PAN_AXIS_X + case.radius, case.lateral, half])
    self.scene = mj_scene.build_cpu_scene(
      mount_height=0.0,
      object_kwargs=dict(kind=case.object_kind, size=case.object_size,
                         height=case.object_height, mass=case.object_mass),
      object_pos=tuple(self.obj_start),
      arm_effort_limit=case.arm_effort,
      gripper_effort_limit=case.gripper_effort,
    )
    sc = self.scene
    m, d = sc.model, sc.data
    # object yaw
    j = sc.jid("object_joint")
    adr = m.jnt_qposadr[j]
    m.qpos0[adr + 3 : adr + 7] = [np.cos(case.object_yaw / 2), 0, 0, np.sin(case.object_yaw / 2)]
    for g in range(m.ngeom):
      if FIXED_JAW.match(m.geom(g).name) or MOVING_JAW.match(m.geom(g).name):
        m.geom_friction[g, 0] = case.finger_friction
    mujoco.mj_resetData(m, d)
    self.arm_adr = np.array([sc.qadr(n) for n in rm.ARM_JOINTS])
    self.arm_act = np.array([sc.aid(n) for n in rm.ARM_JOINTS])
    self.g_adr = sc.qadr(rm.GRIPPER_JOINT)
    self.g_act = sc.aid(rm.GRIPPER_JOINT)
    self.obj_body = sc.bid("object")
    self.obj_geom = sc.gid("object_geom")
    self.site = sc.sid("grasp_site")
    self.home = np.array([0.0, -0.6, 0.4, 1.45, -1.55])
    d.qpos[self.arm_adr] = self.home
    d.qpos[self.g_adr] = 0.6
    d.ctrl[self.arm_act] = self.home
    d.ctrl[self.g_act] = 0.6
    mujoco.mj_forward(m, d)
    self.log: list[dict] = []

  # --- bookkeeping -------------------------------------------------------------
  def object_contacts(self) -> dict:
    m, d = self.scene.model, self.scene.data
    out = {"fixed": 0, "moving": 0, "foreign": 0, "fixed_n": 0.0, "moving_n": 0.0,
           "net_world": np.zeros(3)}
    f6 = np.zeros(6)
    for i in range(d.ncon):
      c = d.contact[i]
      if self.obj_geom not in (c.geom1, c.geom2):
        continue
      other = c.geom2 if c.geom1 == self.obj_geom else c.geom1
      mujoco.mj_contactForce(m, d, i, f6)
      frame = c.frame.reshape(3, 3)  # rows: normal, t1, t2 (normal points geom1->geom2)
      f_world = frame.T @ f6[:3]
      # force on the object: +f if object is geom2, -f if geom1
      out["net_world"] += f_world if c.geom2 == self.obj_geom else -f_world
      name = m.geom(other).name
      if FIXED_JAW.match(name):
        out["fixed"] += 1
        out["fixed_n"] += f6[0]
      elif MOVING_JAW.match(name):
        out["moving"] += 1
        out["moving_n"] += f6[0]
      else:
        out["foreign"] += 1
    return out

  def step(self, n: int = 1, phase: str = "") -> None:
    m, d = self.scene.model, self.scene.data
    for _ in range(n):
      mujoco.mj_step(m, d)
      con = self.object_contacts()
      self.log.append({
        "t": d.time,
        "phase": phase,
        "obj_pos": d.xpos[self.obj_body].copy(),
        "obj_quat": d.xquat[self.obj_body].copy(),
        "site_pos": d.site_xpos[self.site].copy(),
        "grip_R": d.xmat[self.scene.bid("gripper")].reshape(3, 3).copy(),
        "gripper_q": float(d.qpos[self.g_adr]),
        **con,
      })

  def move_to(self, target: np.ndarray, closing: np.ndarray, seconds: float,
              q_hint: np.ndarray, phase: str) -> ik.IkResult:
    m, d = self.scene.model, self.scene.data
    r = ik.multi_start(m, target, closing=closing, q_hint=q_hint, seeds=4,
                       qpos_full=d.qpos.copy())
    q0 = d.ctrl[self.arm_act].copy()
    n = max(1, int(seconds / m.opt.timestep))
    for i in range(n):
      a = (i + 1) / n
      d.ctrl[self.arm_act] = (1 - a) * q0 + a * r.q
      self.step(1, phase)
    return r

  def set_gripper(self, target: float, seconds: float, phase: str) -> None:
    d = self.scene.data
    g0 = float(d.ctrl[self.g_act])
    n = max(1, int(seconds / self.scene.model.opt.timestep))
    for i in range(n):
      d.ctrl[self.g_act] = g0 + (target - g0) * (i + 1) / n
      self.step(1, phase)


def run_pick(case: PickCase) -> PickResult:
  rig = _Rig(case)
  m, d = rig.scene.model, rig.scene.data
  rig.step(100, "settle")
  _, closing_now = mj_scene.gripper_axes(rig.scene)
  closing = _face_normal_closest(case.object_yaw, closing_now)
  obj0 = d.xpos[rig.obj_body].copy()
  half = mj_scene.object_half_height(case.object_kind, case.object_size, case.object_height)
  half_w = case.object_size / 2
  # Grasp-site height: the object centre, but never so low that the lowest fixed-jaw
  # point comes within 3 mm of the table.
  grasp_z = max(half, rm.JAW_TIP_BELOW_SITE + 0.003)
  # Lateral offset: the fixed jaw sits at site + c * (w_nominal / 2) where c is the
  # closing direction. Descend with the fixed jaw 4 mm clear of the object's face.
  clearance = 0.004
  xy = obj0[:2] + closing[:2] * (half_w + clearance - rm.NOMINAL_OBJECT_WIDTH / 2)
  # pre-grasp above the object, jaws open (~72 mm mouth at the moving tip at 0.75 rad)
  r = rig.move_to(np.array([*xy, obj0[2] + 0.07]), closing, 1.5, rig.home, "approach")
  rig.set_gripper(0.75, 0.3, "open")
  for dz in np.linspace(0.055, 0.0, 6):
    r = rig.move_to(np.array([*xy, grasp_z + dz]), closing, 0.25, r.q, "descend")
  rig.step(60, "descend_settle")
  ik_err = r.pos_err
  if case.close_gripper:
    rig.set_gripper(rm.GRIPPER_CLOSED, 1.0, "close")
  rig.step(60, "squeeze")
  gripper_closed_q = float(d.qpos[rig.g_adr])
  for z in np.linspace(grasp_z + 0.01, grasp_z + LIFT_HEIGHT, 8):
    r = rig.move_to(np.array([*xy, z]), closing, 0.2, r.q, "lift")
  hold_start = len(rig.log)
  rig.step(200, "hold")  # 1.0 s
  hold = rig.log[hold_start:]
  open_start = len(rig.log)
  rig.set_gripper(rm.GRIPPER_OPEN, 0.2, "release")
  rig.step(200, "after_release")
  after = rig.log[open_start:]

  table_level = half + 0.003
  # airborne: object centre >= 5 mm above where it started (before release)
  airborne = [e for e in rig.log[:open_start] if e["obj_pos"][2] > obj0[2] + 0.005]
  both = [e for e in airborne if e["fixed"] > 0 and e["moving"] > 0]
  foreign = [e for e in airborne if e["foreign"] > 0]
  # slip: object position in the gripper frame, start vs end of hold
  def in_grip(e):
    return e["grip_R"].T @ (e["obj_pos"] - e["site_pos"])
  slip = float(np.linalg.norm(in_grip(hold[-1]) - in_grip(hold[0]))) if hold else float("nan")
  fell_t = None
  for e in after:
    if e["obj_pos"][2] <= table_level:
      fell_t = e["t"] - after[0]["t"]
      break
  max_z = max(e["obj_pos"][2] for e in rig.log)
  return PickResult(
    case=case.name,
    ik_pos_err_mm=round(ik_err * 1e3, 3),
    object_start_z=round(float(obj0[2]), 4),
    object_max_z=round(float(max_z), 4),
    object_hold_z=round(float(hold[-1]["obj_pos"][2]), 4),
    lifted_by_m=round(float(hold[-1]["obj_pos"][2] - obj0[2]), 4),
    airborne_steps=len(airborne),
    airborne_steps_with_both_jaws=len(both),
    airborne_steps_with_foreign_contact=len(foreign),
    hold_normal_force_fixed_jaw_n=round(float(np.mean([e["fixed_n"] for e in hold])), 3),
    hold_normal_force_moving_jaw_n=round(float(np.mean([e["moving_n"] for e in hold])), 3),
    hold_net_contact_force_z_n=round(float(np.mean([e["net_world"][2] for e in hold])), 4),
    object_weight_n=round(float(case.object_mass * 9.81), 4),
    slip_during_hold_mm=round(slip * 1e3, 3),
    gripper_angle_closed_rad=round(gripper_closed_q, 4),
    object_z_after_open=round(float(after[-1]["obj_pos"][2]), 4),
    fell_within_s=None if fell_t is None else round(float(fell_t), 3),
    finger_contacts_after_open=int(after[-1]["fixed"] + after[-1]["moving"]),
    model_neq=int(m.neq),
    object_is_mocap=bool(m.body_mocapid[rig.obj_body] >= 0),
  )


NOMINAL = PickCase("box30_nominal_r20")
CYL = dict(object_kind="cylinder", object_size=0.030, object_height=0.040, object_mass=0.025)
CASES = [
  NOMINAL,
  PickCase("box30_near_r12", radius=0.12, lateral=-0.02, object_yaw=-0.6),
  PickCase("box30_far_r28", radius=0.28, lateral=0.05, object_yaw=0.9),
  PickCase("box30_weak_servo_low_friction_heavy",
           gripper_effort=rm.STS3215_TORQUE_7V4_6V * 0.5,
           arm_effort=rm.STS3215_TORQUE_7V4_6V, finger_friction=0.5, object_mass=0.05),
  PickCase("box_-15pct", object_size=0.030 * 0.85),
  PickCase("box_+15pct", object_size=0.030 * 1.15),
  # Default task object (30 mm cube) at its +-3 mm DR corners.
  PickCase("box27_far_r27", radius=0.27, lateral=-0.06, object_size=0.027, object_yaw=-1.2),
  PickCase("box33_near_r13", radius=0.13, lateral=0.06, object_size=0.033, object_yaw=2.0),
  # Cylinder option (30 mm x 40 mm) and its DR corners (+-3 mm).
  PickCase("cyl30x40_nominal_r20", **CYL),
  PickCase("cyl_-3mm_far_r27", radius=0.27, lateral=-0.06,
           **{**CYL, "object_size": 0.027, "object_height": 0.037}),
  PickCase("cyl_+3mm_near_r13", radius=0.13, lateral=0.06,
           **{**CYL, "object_size": 0.033, "object_height": 0.043}),
  PickCase("cyl_weak_servo_low_friction_heavy", gripper_effort=rm.STS3215_TORQUE_7V4_6V * 0.5,
           arm_effort=rm.STS3215_TORQUE_7V4_6V, finger_friction=0.4,
           **{**CYL, "object_mass": 0.05}),
]
NEGATIVE = PickCase("negative_control_never_close", close_gripper=False)

_cache: dict[str, PickResult] = {}


def result(case: PickCase) -> PickResult:
  if case.name not in _cache:
    _cache[case.name] = run_pick(case)
  return _cache[case.name]


# ----------------------------------------------------------------------------- tests


def test_model_has_no_weld_latch_or_mocap_object():
  sc = mj_scene.build_cpu_scene(0.0)
  m = sc.model
  assert m.neq == 0, "no equality constraints (weld/connect) allowed in the task model"
  ob = sc.bid("object")
  assert m.body_mocapid[ob] == -1, "object must not be a mocap body"
  j = sc.jid("object_joint")
  assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
  assert m.body_jntnum[ob] == 1
  # no actuator may act on the object
  for a in range(m.nu):
    assert m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
    assert m.actuator_trnid[a, 0] != j


def _assert_physical_pick(res: PickResult, min_lift: float = 0.05) -> None:
  assert res.model_neq == 0 and not res.object_is_mocap
  assert res.lifted_by_m >= min_lift, f"object not lifted: {res}"
  assert res.airborne_steps > 100, res
  # every airborne step: contact with both jaws, and nothing else touches the object
  assert res.airborne_steps_with_both_jaws == res.airborne_steps, res
  assert res.airborne_steps_with_foreign_contact == 0, res
  # contact forces carry the weight (within 10 %)
  assert abs(res.hold_net_contact_force_z_n - res.object_weight_n) < 0.1 * res.object_weight_n, res
  assert res.slip_during_hold_mm < 3.0, res
  # opening the gripper drops the object back to the table
  assert res.fell_within_s is not None and res.fell_within_s < 1.0, res
  assert res.finger_contacts_after_open == 0, res


def test_nominal_pick_is_contact_only_and_drops_on_release():
  _assert_physical_pick(result(NOMINAL))


def test_pick_across_workspace_and_dr_corners():
  for case in CASES[1:]:
    _assert_physical_pick(result(case))


def test_negative_control_open_gripper_never_lifts():
  res = result(NEGATIVE)
  assert res.object_max_z < res.object_start_z + 0.01, res
  assert res.lifted_by_m < 0.005, res


def main() -> None:
  results = [asdict(result(c)) for c in [*CASES, NEGATIVE]]
  for r in results:
    print(json.dumps(r))
  RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
  RESULT_JSON.write_text(json.dumps({"mujoco": mujoco.__version__, "cases": results}, indent=1))
  print(f"wrote {RESULT_JSON}")
  failures = 0
  for name, fn in [
    ("model_has_no_weld", test_model_has_no_weld_latch_or_mocap_object),
    ("nominal", test_nominal_pick_is_contact_only_and_drops_on_release),
    ("workspace_dr", test_pick_across_workspace_and_dr_corners),
    ("negative", test_negative_control_open_gripper_never_lifts),
  ]:
    try:
      fn()
      print(f"PASS {name}")
    except AssertionError as e:
      failures += 1
      print(f"FAIL {name}: {e}")
  sys.exit(1 if failures else 0)


if __name__ == "__main__":
  main()
