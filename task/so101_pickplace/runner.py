"""Runner: mjlab's manipulation runner (checkpoint + ONNX export with mjlab metadata)
plus a JSON sidecar describing the deploy contract (action semantics, observation
layout, task spec) next to every exported ONNX file."""

from __future__ import annotations

import json
import os

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner

from . import TASK_ID
from . import robot_model as rm


def deploy_contract(env) -> dict:
  u = env.unwrapped
  act = u.action_manager.get_term("joint_pos")
  cmd = u.command_manager.get_term("pick_place")
  obs_terms = u.observation_manager.active_terms
  dims = u.observation_manager.group_obs_term_dim
  return {
    "task_id": TASK_ID,
    "control_hz": 1.0 / u.step_dt,
    "physics_dt": u.physics_dt,
    "joint_order": list(act.target_names),
    "action": {
      "kind": "target_delta_joint_position",
      "rule": "target += clip(a, -1, 1) * max_delta; target = clip(target, soft limits)",
      "max_delta_rad": [float(v) for v in act.max_delta.cpu()],
      "initial_target": "measured joint position at episode start",
    },
    "home_joint_pos": {n: float(v) for n, v in zip(
      act.target_names, act._entity.data.default_joint_pos[0, act.target_ids].cpu())},
    "observations": {g: {"terms": t, "dims": [list(d) for d in dims[g]]}
                     for g, t in obs_terms.items()},
    "frames": "robot base frame: x forward, y left, z up, table z=0; pan axis at "
              "(0.0388, 0)",
    "grasp_site_gripper_frame_m": list(rm.GRASP_SITE_POS),
    "reference_duration_s": cmd.ref_duration,
    "notes": [
      "wrist_roll zero: livekit-examples/so-frame measured +90 deg between the "
      "so101_new_calib model zero and the real LeRobot-calibrated arm; verify on the "
      "real arm before the first rollout.",
    ],
  }


class SO101PickPlaceRunner(ManipulationOnPolicyRunner):
  def save(self, path: str, infos=None):
    super().save(path, infos)
    try:
      policy_dir, filename, onnx_path = self._get_export_paths(path)
      with open(os.path.join(str(policy_dir), "deploy_contract.json"), "w",
                encoding="utf-8") as fh:
        json.dump(deploy_contract(self.env), fh, indent=1)
    except Exception as e:  # never kill a long run over metadata
      print(f"[WARN] deploy contract export failed (training continues): {e!r}")
