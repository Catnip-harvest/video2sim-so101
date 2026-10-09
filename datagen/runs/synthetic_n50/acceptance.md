# Acceptance: synthetic

- attempts 51, accepted 50 (**98.0 %**)
- 0.1195 attempts / CPU-second, 0.1171 accepted episodes / CPU-second (render=False)
- accepted episodes: mean 915.3 frames at 30 Hz; in-hand slip {'median': 1.6, 'p90': 3.4, 'max': 5.75} mm; object tilt in hand before lowering {'median': 24.6, 'p90': 28.76, 'max': 37.2} deg; final offset from bin centre {'median': 7.85, 'p90': 11.76, 'max': 18.41} mm
- by servo: {'7v4': {'attempts': 24, 'accepted': 24, 'rate': 1.0}, '12v': {'attempts': 27, 'accepted': 26, 'rate': 0.963}}

| primary rejection reason | count | % of attempts |
|---|---:|---:|
| arm_bin_collision | 1 | 2.0 |

| any rejection reason (an attempt can have several) | count |
|---|---:|
| arm_bin_collision | 1 |
