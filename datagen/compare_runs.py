"""Compare a --wide run's verdicts and staged physics arrays with the narrow run, attempt by attempt."""
import json, sys
from pathlib import Path
import numpy as np
wide, narrow = Path(sys.argv[1]), Path(sys.argv[2])
def rows(d):
  return {json.loads(l)["attempt"]: json.loads(l) for l in (d/"attempts.jsonl").read_text().splitlines() if l.strip()}
rw, rn = rows(wide), rows(narrow)
common = sorted(set(rw) & set(rn))
verdict_diff = [i for i in common if rw[i]["accepted"] != rn[i]["accepted"] or rw[i]["reasons"] != rn[i]["reasons"]]
n_arr, arr_diff, missing = 0, [], 0
for i in common:
  if not rw[i]["accepted"]:
    continue
  a, b = wide/"staging"/f"attempt_{i:06d}.npz", narrow/"staging"/f"attempt_{i:06d}.npz"
  if not (a.exists() and b.exists()):
    missing += 1; continue
  za, zb = np.load(a), np.load(b)
  n_arr += 1
  if za.files != zb.files or not all(np.array_equal(za[k], zb[k]) for k in za.files):
    arr_diff.append(i)
print(json.dumps({"attempts_compared": len(common), "verdict_differences": len(verdict_diff),
                  "verdict_diff_ids": verdict_diff[:20], "accepted_npz_compared": n_arr,
                  "npz_array_differences": len(arr_diff), "diff_ids": arr_diff[:20], "missing_npz": missing}))
