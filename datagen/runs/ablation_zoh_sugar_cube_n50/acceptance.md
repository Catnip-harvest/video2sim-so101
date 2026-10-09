# Acceptance: pexels_7488188_sugar_cube

- attempts 69, accepted 50 (**72.5 %**)
- 0.3544 attempts / CPU-second, 0.2568 accepted episodes / CPU-second (render=False)
- accepted episodes: mean 539.9 frames at 30 Hz; in-hand slip {'median': 9.82, 'p90': 13.16, 'max': 15.77} mm; object tilt in hand before lowering {'median': 1.6, 'p90': 8.18, 'max': 9.7} deg; final offset from bin centre {'median': 1.66, 'p90': 4.27, 'max': 7.36} mm
- by servo: {'7v4': {'attempts': 37, 'accepted': 35, 'rate': 0.9459}, '12v': {'attempts': 32, 'accepted': 15, 'rate': 0.4688}}

| primary rejection reason | count | % of attempts |
|---|---:|---:|
| dropped_or_slipped | 19 | 27.5 |

| any rejection reason (an attempt can have several) | count |
|---|---:|
| dropped_or_slipped | 19 |
| foreign_contact_while_carried | 15 |
| joint_limit | 7 |
| ik_unreachable | 6 |
