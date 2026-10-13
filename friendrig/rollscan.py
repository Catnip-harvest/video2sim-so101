"""Scan the wrist_roll offset (and a coarse elbow/wrist_flex grid) with session cameras frozen."""
import sys, json, numpy as np, multiprocessing as mp, os
sys.path.insert(0, '/work/wt-friendrig')
from friendrig import camfit as C
out = '/work/outputs_large/friendrig/camfit'
src = sys.argv[1]
cs = json.load(open(f'{out}/{src}'))
base = np.array(cs.get('offsets_deg_added_to_default', [0]*5))
z = C.load_frames(out)
rng = np.random.default_rng(1)
items = []
for s in cs['sessions']:
  ix = np.where(np.isin(z['episode'], s['train_episodes']) & (z['split'] == 'train'))[0]
  ix = rng.choice(ix, min(25, len(ix)), replace=False)
  items.append((z['state'][ix], [C.lowres(z['arm'][i], z['ignore'][i]) for i in ix], [s['camera'][k] for k in C.CAM_NAMES]))
def job(off):
  C._G.setdefault('pa', C.PointArm(20000))
  tot = []
  for st, re, p in items:
    tot += C.Scorer(C._G['pa'], st, re, offsets=np.asarray(off)).per_frame(p)
  return float(np.mean(tot))
cands = []
for wr in range(-180, 180, 30):
  o = base.copy(); o[4] = wr - (-90) if False else wr; cands.append(o.copy())
with mp.get_context('fork').Pool(8) as pool:
  res = pool.map(job, cands)
for o, r in zip(cands, res): print('offsets', np.round(o, 1).tolist(), 'mean soft IoU', round(r, 4), flush=True)
