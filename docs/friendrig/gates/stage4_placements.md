# Stage 4 gate: placement distribution — PASS

**Detection rate 208 / 220 = 94.5 %** of TRAIN table-to-bowl episodes (regex on the task text in
`friendrig/objects.py`; 220 of 474 train episodes). Not detected: 10 with two green blobs (multi-cube), 1 with no
white blob, 1 low fit IoU.

## Distribution (robot base frame, metres; per-episode table `placements_train.csv` on the server, NOT in git)
| | x mean (sd) | y mean (sd) | x p5-p95 | y p5-p95 |
|---|---|---|---|---|
| cube start | 0.200 (0.047) | 0.134 (0.094) | 0.131-0.271 | 0.003-0.271 |
| bowl | 0.248 (0.038) | 0.166 (0.052) | 0.194-0.310 | 0.075-0.236 |

Cube-to-bowl distance median 0.165 m (p5-p95 0.136-0.199). Everything is on the robot's left (y > 0), where the
camera looks. Bowls come in clusters (the bowl stays put for runs of episodes). Plot: `placements_train.png`
(server/Drive).

## Sim sampler (`friend_bowl` profile)
Draw a real (cube, bowl) pair at random from the table, add Gaussian jitter **sd 15 mm** to each (the stated
widening), cube yaw uniform +-45 deg, and redraw until both are inside the engine's reach band (0.08-0.30 m from
the pan axis), the **cube is within 0.28 m** (the scripted expert cannot lift the cube to rim height from further
out: 3/20 ik_unreachable in the first smoke run, all at r ~0.29 m), and the cube-bowl gap is >= rim radius +
0.75 cube + 1 cm.

**Honest limit:** only 117 / 208 = 56 % of the measured pairs are inside that band before jitter (43 % of real bowls
sit beyond 0.28 m from the pan axis, p95 0.333 m). The generated scenes over-represent the nearer part of the real
layouts; the human operator reached further, with a tilted gripper.

## Command
`python -m friendrig.objects fit --out /work/outputs_large/friendrig/camfit --cams cam_sessions_offsets_roll.json --workers 8`
(log `/root/work/logs/friendrig/objects.log`, output `objects.json`, `placements_train.csv`, `placements_train.png`).
