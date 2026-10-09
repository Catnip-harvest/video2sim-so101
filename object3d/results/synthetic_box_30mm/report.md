### Settle: PASS

| height err mm | penetration mm | speed mm/s (last 0.5 s) | spin deg/s | drift mm (last 1 s) | tilt deg | settled after s |
|---:|---:|---:|---:|---:|---:|---:|
| -0.106 | 0.108 | 0.352 | 0.888 | 0.001 | 0.005 | 0.133 |

### Parity: A = box:0.030:0.025, B = synthetic_box_30mm (seed 0, same draws)

| | grasp + lift | carried, no drop | full episode accepted | slip mm median/p90 | lift mm median | top rejections |
|---|---|---|---|---|---|---|
| box:0.030:0.025 | 30/30 (100 %, CI 89-100) | 30/30 (100 %, CI 89-100) | 30/30 (100 %, CI 89-100) | 0.2/0.3 | 60.08 | {} |
| synthetic_box_30mm | 30/30 (100 %, CI 89-100) | 30/30 (100 %, CI 89-100) | 30/30 (100 %, CI 89-100) | 0.16/0.28 | 60.0 | {} |

Paired (accepted): both 30, only A 0, only B 0, neither 0; A - B = 0 points.
