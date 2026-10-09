"""Opt-in runtime switches for AMD ROCm (MI300X); all OFF unless the env var is set.

MJLAB_DISABLE_CUDA_GRAPH=1  ROCm Warp reports is_cuda, so mjlab would try CUDA-graph
                            capture, which the ROCm port does not support. Forces the
                            eager path.
MJLAB_GRAPH_CONDITIONAL=0   mjwarp Option.graph_conditional = False (needs CUDA 12.4
                            conditional graph nodes). Reported needed in
                            AMD-Ecosystem/warp#20.
Idempotent. This task has no cameras, so no texture switch is needed.
"""

from __future__ import annotations

import functools
import os

_DONE: set[str] = set()


def apply() -> None:
  if os.environ.get("MJLAB_DISABLE_CUDA_GRAPH", "") in ("1", "true") and "graph" not in _DONE:
    from mjlab.sim import sim as mjlab_sim

    mjlab_sim.Simulation._should_use_cuda_graph = lambda self: False  # type: ignore[method-assign]
    _DONE.add("graph")
    print("[so101_pickplace] CUDA/HIP graph capture disabled")
  if os.environ.get("MJLAB_GRAPH_CONDITIONAL", "") == "0" and "cond" not in _DONE:
    from mjlab.sim import sim as mjlab_sim

    original = mjlab_sim.SimulationCfg.apply_wp_opt

    @functools.wraps(original)
    def apply_wp_opt(self, wp_opt) -> None:
      original(self, wp_opt)
      if hasattr(wp_opt, "graph_conditional"):
        wp_opt.graph_conditional = False

    mjlab_sim.SimulationCfg.apply_wp_opt = apply_wp_opt  # type: ignore[method-assign]
    _DONE.add("cond")
    print("[so101_pickplace] mjwarp graph_conditional forced False")
