"""Plot the human demo's object path next to the generated episodes' object paths.

Works on a no-render dataset (uses observation.environment_state), so it is the
portfolio figure that can be made on the laptop.

  python plot_trajectories.py --dataset runs/sugar/lerobot --spec SPEC.json --out traj.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

from r2s.paths import load_task_spec, mj_scene  # noqa: E402
from validate_dataset import read_parquet  # noqa: E402


def load_episodes(root: Path, max_eps: int):
  data = {}
  for f in sorted((root / "data").glob("*/*.parquet")):
    cols, _ = read_parquet(f)
    for k in ("episode_index", "observation.environment_state", "timestamp"):
      data.setdefault(k, []).append(np.asarray(cols[k]))
  ep = np.concatenate(data["episode_index"])
  env = np.concatenate(data["observation.environment_state"]).astype(float)
  meta = [json.loads(line) for line in
          (root / "meta" / "datagen_episodes.jsonl").read_text().splitlines() if line.strip()]
  out = []
  for e in range(min(max_eps, int(ep.max()) + 1)):
    out.append((env[ep == e, :3], meta[e]))
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--dataset", required=True)
  ap.add_argument("--spec", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--max-episodes", type=int, default=50)
  args = ap.parse_args()
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.patches import Circle, Rectangle

  spec = load_task_spec(args.spec)
  eps = load_episodes(Path(args.dataset), args.max_episodes)
  fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
  cmap = plt.get_cmap("viridis")
  for k, (p, meta) in enumerate(eps):
    col = cmap(k / max(len(eps) - 1, 1))
    ax0.plot(p[:, 1], p[:, 0], color=col, lw=0.8, alpha=0.6)
    bx, by = meta["variation"]["bin_xy"]
    h = mj_scene.BIN_INNER_HALF
    ax0.add_patch(Rectangle((by - h, bx - h), 2 * h, 2 * h, fill=False, ec=col, lw=0.4, alpha=0.4))
    ps = meta["phase_starts"]
    a, b = ps.get("lift", 0), ps.get("release", len(p) - 1)
    s = np.linspace(0, 1, b - a + 1)
    ax1.plot(s, p[a:b + 1, 2] * 1e3, color=col, lw=0.8, alpha=0.6)
  if spec.trajectory is not None:
    wp = np.asarray(spec.trajectory.waypoints)
    ax0.plot(wp[:, 1], wp[:, 0], "r-", lw=2.5, label="human demo object path (spec)")
    t, pos, g, r = spec.trajectory.reference()
    ig, ir = int(np.searchsorted(t, g)), min(int(np.searchsorted(t, r)), len(t) - 1)
    s = np.linspace(0, 1, ir - ig + 1)
    ax1.plot(s, pos[ig:ir + 1, 2] * 1e3, "r-", lw=2.5, label="human demo (grasp to release)")
  ax0.add_patch(Circle((0, 0.0388), 0.03, color="0.3", alpha=0.5))
  ax0.text(0, 0.0388, "SO-101\nbase", ha="center", va="center", fontsize=7, color="w")
  ax0.set_xlabel("y (m, robot left +)")
  ax0.set_ylabel("x (m, forward)")
  ax0.invert_xaxis()
  ax0.set_aspect("equal")
  ax0.set_title(f"Top view: {len(eps)} generated episodes (object centre)")
  ax0.legend(loc="lower left", fontsize=8)
  ax1.set_xlabel("progress, lift-off to release (normalised)")
  ax1.set_ylabel("object centre height (mm)")
  ax1.set_title("Height profile while carried")
  ax1.legend(fontsize=8)
  fig.suptitle(f"Real-to-sim data engine: {Path(args.spec).parent.name}", fontsize=11)
  fig.tight_layout()
  fig.savefig(args.out, dpi=110)
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
