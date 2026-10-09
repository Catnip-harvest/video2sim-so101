"""Which MuJoCo visual setting makes OSMesa slow? Renders 40 frames x 2 cams per variant."""
import json, sys, time
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2s.paths import load_task_spec
from r2s.variations import sample_variation
from r2s.scene import build_episode_scene

run = Path(sys.argv[1]); cfg = json.loads((run / "run.json").read_text())
spec = load_task_spec(cfg["spec"])
stem = sorted((run / "staging").glob("attempt_*.npz"))[0]
v = sample_variation(spec, cfg["seed"], int(stem.stem.split("_")[1]))
q = np.load(stem)["qpos_full"][:40]
sc = build_episode_scene(v, object_kind=spec.object.kind, render_size=(256, 256))
m = sc.model
print("shadowsize", m.vis.quality.shadowsize, "offsamples", m.vis.quality.offsamples,
      "nmeshvert", m.nmeshvert, "nmeshface", m.nmeshface, "nlight", m.nlight, "ngeom", m.ngeom,
      "off", m.vis.global_.offwidth, m.vis.global_.offheight)

def bench(name, setup=lambda m: None, flags={}):
  m = build_episode_scene(v, object_kind=spec.object.kind, render_size=(256, 256)).model
  setup(m)
  d = mujoco.MjData(m); r = mujoco.Renderer(m, 256, 256)
  for f, val in flags.items():
    r.scene.flags[f] = val
  t0 = time.perf_counter(); n = 0
  for x in q:
    d.qpos[:] = x; mujoco.mj_forward(m, d)
    for cam in ("front", "wrist_cam"):
      r.update_scene(d, camera=cam)
      for f, val in flags.items():
        r.scene.flags[f] = val
      r.render(); n += 1
  dt = time.perf_counter() - t0; r.close()
  print(f"{name:30s} {n/dt:6.1f} images/s")

R=mujoco.mjtRndFlag
def a(m): m.vis.quality.offsamples=0; m.vis.quality.shadowsize=1024
bench("offs0+shadow1024+norefl", a, {R.mjRND_REFLECTION:0})
def b(m): m.vis.quality.offsamples=0; m.vis.quality.shadowsize=512
bench("offs0+shadow512+norefl", b, {R.mjRND_REFLECTION:0})
