# Acceptance: synthetic

- attempts 56, accepted 50 (**89.3 %**)
- 0.0889 attempts / CPU-second, 0.0794 accepted episodes / CPU-second (render=False)
- accepted episodes: mean 919.2 frames at 30 Hz; in-hand slip {'median': 8.8, 'p90': 12.66, 'max': 15.7} mm; object tilt in hand before lowering {'median': 22.7, 'p90': 34.45, 'max': 37.3} deg; final offset from bin centre {'median': 7.17, 'p90': 10.93, 'max': 25.35} mm
- by servo: {'7v4': {'attempts': 26, 'accepted': 25, 'rate': 0.9615}, '12v': {'attempts': 30, 'accepted': 25, 'rate': 0.8333}}

| primary rejection reason | count | % of attempts |
|---|---:|---:|
| dropped_or_slipped | 4 | 7.1 |
| foreign_contact_while_carried | 1 | 1.8 |
| self_collision | 1 | 1.8 |

| any rejection reason (an attempt can have several) | count |
|---|---:|
| foreign_contact_while_carried | 5 |
| dropped_or_slipped | 4 |
| ik_unreachable | 1 |
| self_collision | 1 |
| not_in_place_region | 1 |
