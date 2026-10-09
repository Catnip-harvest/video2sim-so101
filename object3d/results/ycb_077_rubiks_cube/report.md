### Settle: FAIL spin_deg_s

| height err mm | penetration mm | speed mm/s (last 0.5 s) | spin deg/s | drift mm (last 1 s) | tilt deg | settled after s |
|---:|---:|---:|---:|---:|---:|---:|
| -0.187 | 0.207 | 1.935 | 3.843 | 0.002 | 0.128 | 0.133 |

### Parity: A = box:0.057:0.094, B = ycb_077_rubiks_cube (seed 0, same draws)

| | grasp + lift | carried, no drop | full episode accepted | slip mm median/p90 | lift mm median | top rejections |
|---|---|---|---|---|---|---|
| box:0.057:0.094 | 11/30 (37 %, CI 22-55) | 11/30 (37 %, CI 22-55) | 10/30 (33 %, CI 19-51) | 8.23/10.32 | 54.91 | {'grasp_failed': 19, 'ik_unreachable': 1} |
| ycb_077_rubiks_cube | 15/30 (50 %, CI 33-67) | 15/30 (50 %, CI 33-67) | 15/30 (50 %, CI 33-67) | 5.73/9.3 | 56.05 | {'grasp_failed': 13, 'dropped_or_slipped': 2} |

Paired (accepted): both 10, only A 0, only B 5, neither 15; A - B = -17 points.

### Scale sensitivity: ycb_077_rubiks_cube

| scale | grasp + lift | full episode accepted | slip mm median | top rejections |
|---|---|---|---|---|
| x1 | 15/30 (50 %, CI 33-67) | 15/30 (50 %, CI 33-67) | 5.73 | {'grasp_failed': 13, 'dropped_or_slipped': 2} |
| x0.95 | 17/30 (57 %, CI 39-73) | 14/30 (47 %, CI 30-64) | 5.99 | {'grasp_failed': 13, 'torque_saturation': 2, 'dropped_or_slipped': 1} |
| x0.97 | 17/30 (57 %, CI 39-73) | 16/30 (53 %, CI 36-70) | 6.0 | {'grasp_failed': 13, 'torque_saturation': 1} |
| x1.03 | 10/30 (33 %, CI 19-51) | 10/30 (33 %, CI 19-51) | 8.02 | {'grasp_failed': 20} |
| x1.05 | 5/30 (17 %, CI 7-34) | 4/30 (13 %, CI 5-30) | 8.26 | {'grasp_failed': 25, 'dropped_or_slipped': 1} |
