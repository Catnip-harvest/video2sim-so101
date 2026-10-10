# Branch `jepa/judge`: can a video model (V-JEPA 2) judge success? (Result: no, as specified)

## What this branch does
- Builds a **judge set** of 775 sim episodes (509 passed, 266 failed). Labels come only from the physics verifier. Failures include natural arm-table touches, zero-order-hold drops, a jaw that never closes, and an early release (`generate.py --inject {no_close,early_release}`, `--widen`). No welds, no teleports: injections change only the expert's jaw commands.
- Splits are by scene (all episodes of a scene stay together): `train` 394, `test` 235, `cam_test` 146 (raised camera).
- Encodes each episode with frozen `facebook/vjepa2-vitl-fpc64-256` (16 frames) and trains a logistic probe. Compares it with a **last-frame** baseline (DINOv2 or V-JEPA on the last frame only).
- Pre-registered gate: probe AUC must beat the best last-frame AUC by >= 0.02 with paired DeLong p < 0.05.
- Also a **realism score**: Frechet distance (FD) between sim clips and the friend's real clips in V-JEPA space.

## How to run (inside container `sprint`)
```
bash /work/wt-jepa/jepa/gen_negatives.sh          # CPU, 6 workers, nice 19, ~38 min
cd /work/wt-jepa
python jepa/build_judge_set.py                    # -> judge_set.json
python jepa/extract.py sim                        # GPU ~13 min, ~1.2 GB; also: real, wrist_last
python jepa/probe.py                              # CPU -> probe_results.json
python jepa/realism.py                            # CPU -> realism_results.json
```
Data and features: `/work/outputs_large/jepa/`. Logs: `/root/work/logs/jepa_{gen,extract,realism,probe}.log`.

## Results (source: `/root/work/gates/gate_jepa.md`)
**Verdict: FAIL. JEPA lost.**

| Contender (y_full, `test` split, 164 pos / 71 neg) | AUC [95 % DeLong CI] |
|---|---|
| V-JEPA probe (the gated one) | **0.784 [0.720, 0.849]** |
| last frame, DINOv2 (best baseline) | **0.802 [0.737, 0.868]** |
| last frame, V-JEPA | 0.799 [0.736, 0.861] |
| shuffled labels, 20 runs | mean 0.512 |

- Gate: probe minus last-frame = **-0.018**, CI [-0.084, +0.048], p = 0.59. Needed +0.02.
- Shuffled labels sit near 0.5, so the pipeline does not leak labels.
- Not gated, picked **after** seeing results (hypotheses only, need a fresh split): keeping the 8 time slots in order 0.866; wrist-camera video 0.912 (vs wrist last frame 0.798).

Realism (FD in V-JEPA space, 200 vs 200 clips, 20 splits):
- real vs real floor **15.0**, sim vs sim floor 13.2, **sim vs real 1328.8** (about 89x the floor). A classifier separates sim from real perfectly (AUC 1.000).
- The gap is not camera height (1332 / 1335 / 1341 by tercile). It is viewpoint, scene content and the flat fast renders.

Limits: the cube is ~6 px in a 256 px front render (smaller than one V-JEPA patch); only 71 test negatives (CI about +-0.065).

## What you may / may not claim (honest-claims list)
May claim:
- A pre-registered test found the V-JEPA probe **no better** than a last-frame baseline on this sim judge set.
- Our sim renders are far from the real clips in V-JEPA space (FD 1329 vs floor 15).

May not claim:
- **That the JEPA judge works** (0.784 vs 0.802, FAIL).
- That the time-concat or wrist-video variants work: they were chosen after looking.
- Anything about a real robot (G0 overlay INCONCLUSIVE; sim-to-real gap FD 1329 vs floor 15).
- A difference between models without a significant paired test; laptop speed; any drawer, C/D, Hanoi or dexterous result.
