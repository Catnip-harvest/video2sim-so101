"""Scattered-cube stacking: rules, next-move solver, spawn zone and per-episode draws.

The task. The Hanoi cubes (large 45 mm blue, medium 35 mm green, small 25 mm red) lie
anywhere in a spawn zone on the table, at any yaw, with no spot markers. The goal is ONE
tower where the large cube lies: large at the bottom, medium on large, small on medium.
One episode = one correct next move. The instruction never changes:
"stack the cubes into a tower with the biggest at the bottom".

State. Each cube is on the table ("T") or on one other cube. A state is three letters, the
support of (large, medium, small): "TTT" = all on the table, "TLM" = the goal (medium on
large, small on medium), "TTL" = small on large, medium alone. With three cubes there are
13 states (all on the table; 6 two-cube stacks + a single cube; 6 three-cube stacks).

Next-move rule (``next_move``), as specified for this dataset:
  1. a cube that is NOT on the table and NOT part of the correct tower (large on table,
     medium on that large, small on that medium) is "wrongly placed"; if one exists, the
     wrongly placed cube with nothing on top goes to free table space;
  2. otherwise build: medium onto large, then small onto medium.
A breadth-first search over every legal move (any clear cube onto the table or onto any
other clear cube) gives the true minimum number of moves; the rule matches it in every
state except where the medium cube sits on the small one, where the rule spends one extra
move (it puts the medium cube on the table first instead of directly onto the large one).
Tests check the rule reaches the goal from all 13 states.

Spawn zone: a polar sector around the shoulder-pan axis, r in ZONE_R, azimuth within
+-ZONE_AZ_DEG. ZONE_R comes from the reachability sweep in README_stack.md: the arm must
grasp at table height AND place on a two-cube stack anywhere in the zone, then lift its
open jaws 20 mm clear of the finished tower (grasp site ~124 mm up). The measured reach
ceiling at the 20 deg finger tilt is 120 mm at r = 0.14 m and 128 mm at 0.15 m, so the zone
starts at 0.15 m.

Start buckets (``sample_stack_variation``):
  scatter  50 %  all three on the table; strata: default (gap >= 20 mm between footprint
                 circles, 70 %), near (a tight cluster, gaps 3-10 mm, 15 %), far_near (one
                 cube at r >= 0.85 r_max, the other two 3-10 mm apart, 15 %)
  partial  25 %  medium on large, small elsewhere on the table ("TLT")
  wrong    17 %  a physically stable wrong stack (small on large / on medium, medium on
                 small, large on medium / on small, or medium on small on large); the
                 runner settles it and rejects the attempt if it collapses
  noise     8 %  a start drawn from the three buckets above, executed by the expert with
                 Gaussian joint-target noise (MimicGen-style); kept only if it passes
Footprint circle of a cube = edge * 0.707 (half its diagonal); separations are between
those circles. Every invalid draw is redrawn; after MAX_TRIES the sampler raises.
"""

from __future__ import annotations

import copy
import itertools
import math
from collections import deque

import numpy as np

from .hanoi import CUBE_NAMES, CUBES, MASS_RANGE, SIZES, _cube_rgb
from .paths import TaskSpec
from .variations import PAN_AXIS, Variation, sample_variation

INSTRUCTION = "stack the cubes into a tower with the biggest at the bottom"
TABLE = "T"
GOAL_STATE = "TLM"
ALL_ON_TABLE = "TTT"
PARTIAL_STATE = "TLT"

# ---- spawn zone (see the reachability sweep in README_stack.md) ----------------------------
ZONE_R = (0.15, 0.24)  # m from the shoulder-pan axis
ZONE_AZ_DEG = 60.0
FOOTPRINT = {c: SIZES[c] * 0.707 for c in CUBES}  # footprint circle radius (m)
GAP_DEFAULT = 0.020
GAP_NEAR = (0.003, 0.010)
FAR_FRACTION = 0.85
MAX_TRIES = 5000
SPOT_MARGIN = 0.010  # a free spot for an unstacked cube stays 1 cm inside the zone
CUBE_ON_CUBE_XY = 0.003  # a stacked cube's centre within this of the one below (per axis)
START_GAP = 0.0004  # m per stack level: dropped into place from just above, then settled

BUCKETS = {"scatter": 0.50, "partial": 0.25, "wrong": 0.17, "noise": 0.08}
SCATTER_STRATA = {"default": 0.70, "near": 0.15, "far_near": 0.15}
WRONG_STATES = {  # supports of (L, M, S) -> weight; all physically stable when centred
  "TTL": 1.0,   # small on large
  "TTM": 1.0,   # small on medium
  "TST": 1.0,   # medium on small
  "MTT": 0.5,   # large on medium
  "STT": 0.5,   # large on small
  "TSL": 0.5,   # medium on small on large
}
NOISE = {"sigma_rad": 0.012, "tau_s": 0.25}  # arm joint-target noise (noise bucket)


# ---- states and moves ------------------------------------------------------------------------
def supports(state: str) -> dict[str, str]:
  return dict(zip(CUBES, state))


def is_valid(state: str) -> bool:
  if len(state) != 3 or any(x not in (TABLE,) + CUBES for x in state):
    return False
  sup = supports(state)
  if any(sup[c] == c for c in CUBES):
    return False
  on = [s for s in sup.values() if s != TABLE]
  if len(on) != len(set(on)):  # two cubes on the same cube
    return False
  for c in CUBES:  # no cycles: following supports must reach the table
    seen, x = set(), c
    while x != TABLE:
      if x in seen:
        return False
      seen.add(x)
      x = sup[x]
  return True


def all_states() -> list[str]:
  return ["".join(p) for p in itertools.product((TABLE,) + CUBES, repeat=3) if is_valid("".join(p))]


def on_top_of(state: str, cube: str) -> str | None:
  for c, s in supports(state).items():
    if s == cube:
      return c
  return None


def is_clear(state: str, cube: str) -> bool:
  return on_top_of(state, cube) is None


def towers(state: str) -> list[list[str]]:
  """Every stack (single cubes included), bottom to top."""
  out = []
  for c in CUBES:
    if supports(state)[c] == TABLE:
      t = [c]
      while (up := on_top_of(state, t[-1])) is not None:
        t.append(up)
      out.append(t)
  return out


def below_count(state: str, cube: str) -> int:
  sup, n, x = supports(state), 0, cube
  while sup[x] != TABLE:
    x, n = sup[x], n + 1
  return n


def legal_moves(state: str) -> list[tuple[str, str]]:
  """(cube, target): a clear cube onto the table or onto another clear cube."""
  sup, moves = supports(state), []
  for c in CUBES:
    if not is_clear(state, c):
      continue
    if sup[c] != TABLE:
      moves.append((c, TABLE))
    for x in CUBES:
      if x != c and is_clear(state, x) and sup[c] != x:
        moves.append((c, x))
  return moves


def apply_move(state: str, move: tuple[str, str]) -> str:
  if move not in legal_moves(state):
    raise ValueError(f"illegal move {move} from {state}")
  cube, target = move
  i = CUBES.index(cube)
  return state[:i] + target + state[i + 1:]


def correctly_placed(state: str, cube: str) -> bool:
  sup = supports(state)
  if cube == "L":
    return sup["L"] == TABLE
  if cube == "M":
    return sup["M"] == "L" and sup["L"] == TABLE
  return sup["S"] == "M" and correctly_placed(state, "M")


def wrongly_placed(state: str) -> list[str]:
  return [c for c in CUBES if supports(state)[c] != TABLE and not correctly_placed(state, c)]


def next_move(state: str) -> tuple[str, str] | None:
  """The dataset's next move (module docstring), or None at the goal."""
  if state == GOAL_STATE:
    return None
  wrong = [c for c in wrongly_placed(state) if is_clear(state, c)]
  if wrong:
    return (wrong[0], TABLE)  # with three cubes at most one wrong cube is clear
  sup = supports(state)
  if not (sup["M"] == "L" and sup["L"] == TABLE):
    return ("M", "L")
  return ("S", "M")


def _distances_to_goal() -> dict[str, int]:
  dist, queue = {GOAL_STATE: 0}, deque([GOAL_STATE])
  while queue:
    s = queue.popleft()
    for mv in legal_moves(s):  # every move can be undone, so search outward from the goal
      t = apply_move(s, mv)
      if t not in dist:
        dist[t] = dist[s] + 1
        queue.append(t)
  return dist


OPTIMAL_MOVES = _distances_to_goal()


def rule_moves_to_goal(state: str) -> list[tuple[str, str]]:
  out = []
  while state != GOAL_STATE:
    mv = next_move(state)
    out.append(mv)
    state = apply_move(state, mv)
    if len(out) > 10:
      raise RuntimeError("rule does not converge")
  return out


def centre_height(state: str, cube: str) -> float:
  sup, z, x = supports(state), SIZES[cube] / 2, cube
  while sup[x] != TABLE:
    x = sup[x]
    z += SIZES[x]
  return z


def describe(state: str) -> str:
  parts = []
  for t in towers(state):
    parts.append(" on ".join(CUBE_NAMES[c] for c in reversed(t)) if len(t) > 1 else CUBE_NAMES[t[0]])
  return "; ".join(parts)


# ---- spawn zone geometry ---------------------------------------------------------------------
def polar(xy) -> tuple[float, float]:
  d = np.asarray(xy, float)[:2] - PAN_AXIS
  return float(np.linalg.norm(d)), math.degrees(math.atan2(d[1], d[0]))


def in_zone(xy, margin: float = 0.0) -> bool:
  """Inside the sector, at least ``margin`` (m) from its edges."""
  r, az = polar(xy)
  az_margin = math.degrees(margin / max(r, 1e-6))
  return ZONE_R[0] + margin <= r <= ZONE_R[1] - margin and abs(az) <= ZONE_AZ_DEG - az_margin


def _sample_zone_point(rng: np.random.Generator, r_min: float | None = None) -> np.ndarray:
  lo = ZONE_R[0] if r_min is None else r_min
  r = math.sqrt(rng.uniform(lo ** 2, ZONE_R[1] ** 2))  # uniform over the sector's area
  a = math.radians(rng.uniform(-ZONE_AZ_DEG, ZONE_AZ_DEG))
  return PAN_AXIS + r * np.array([math.cos(a), math.sin(a)])


def _base_radius(tower: list[str]) -> float:
  return max(FOOTPRINT[c] for c in tower)  # a larger cube on a smaller one overhangs


def free_spot(rng: np.random.Generator, cube: str, others: list[tuple[np.ndarray, float]],
              gap: float = GAP_DEFAULT) -> np.ndarray:
  """A table spot in the zone whose footprint circle (cube) is >= gap from every other
  footprint circle in ``others`` [(centre xy, radius)]."""
  for _ in range(MAX_TRIES):
    p = _sample_zone_point(rng)
    if not in_zone(p, SPOT_MARGIN):
      continue
    if all(np.linalg.norm(p - c) >= FOOTPRINT[cube] + r + gap for c, r in others):
      return p
  raise RuntimeError(f"no free spot for the {CUBE_NAMES[cube]} cube in the zone")


def _place_towers(rng: np.random.Generator, tws: list[list[str]], stratum: str) -> list[np.ndarray]:
  """Centres (xy) of each tower's base, by stratum. Raises after MAX_TRIES invalid draws."""
  rad = [_base_radius(t) for t in tws]
  n = len(tws)
  for _ in range(MAX_TRIES):
    pts: list[np.ndarray] = []
    if stratum == "far_near" and n == 3:
      far = int(rng.integers(n))
      near = [k for k in range(n) if k != far]
      pts_k = {far: _sample_zone_point(rng, FAR_FRACTION * ZONE_R[1])}
      a = _sample_zone_point(rng)
      ang = rng.uniform(-math.pi, math.pi)
      g = rng.uniform(*GAP_NEAR)
      b = a + (rad[near[0]] + rad[near[1]] + g) * np.array([math.cos(ang), math.sin(ang)])
      pts_k[near[0]], pts_k[near[1]] = a, b
      pts = [pts_k[k] for k in range(n)]
      need = {(near[0], near[1]): GAP_NEAR[0], (near[1], near[0]): GAP_NEAR[0]}
    elif stratum == "near" and n == 3:
      a = _sample_zone_point(rng)
      ang = rng.uniform(-math.pi, math.pi)
      b = a + (rad[0] + rad[1] + rng.uniform(*GAP_NEAR)) * np.array([math.cos(ang), math.sin(ang)])
      k = int(rng.integers(2))  # the third cube hugs the first or the second
      base = (a, b)[k]
      ang3 = rng.uniform(-math.pi, math.pi)
      c = base + (rad[k] + rad[2] + rng.uniform(*GAP_NEAR)) * np.array([math.cos(ang3), math.sin(ang3)])
      pts = [a, b, c]
      need = {(i, j): GAP_NEAR[0] for i in range(3) for j in range(3) if i != j}
    else:
      pts = [_sample_zone_point(rng) for _ in range(n)]
      need = {}
    ok = all(in_zone(p) for p in pts)
    for i, j in itertools.combinations(range(n), 2):
      g = need.get((i, j), GAP_DEFAULT)
      if np.linalg.norm(pts[i] - pts[j]) < rad[i] + rad[j] + g - 1e-9:
        ok = False
    if ok:
      return pts
  raise RuntimeError(f"could not place {n} towers for stratum {stratum!r}")


def _draw(rng: np.random.Generator, weights: dict) -> str:
  keys = list(weights)
  p = np.array([weights[k] for k in keys], float)
  return keys[int(rng.choice(len(keys), p=p / p.sum()))]


# ---- per-episode draw ----------------------------------------------------------------------------
def stack_info(v: Variation) -> dict | None:
  return (v.extras or {}).get("stack")


def sample_stack_variation(spec: TaskSpec, base_seed: int, attempt: int, variety: bool = True,
                           bucket: str | None = None, stratum: str | None = None,
                           start_state: str | None = None) -> Variation:
  """One single-move stacking episode: the normal Variation (servos, look, camera, expert
  style), re-laid-out from a separate random stream ([seed, 17]). ``bucket`` / ``stratum``
  / ``start_state`` force a draw (eval rows and tests); None draws them from the mix."""
  v = sample_variation(spec, base_seed, attempt, variety=variety)
  rng = np.random.default_rng([v.seed, 17])
  drawn_bucket = _draw(rng, BUCKETS)
  bucket = bucket or drawn_bucket
  base_bucket = bucket
  if bucket == "noise":
    base_bucket = _draw(rng, {k: w for k, w in BUCKETS.items() if k != "noise"})
  drawn_stratum = _draw(rng, SCATTER_STRATA)
  drawn_wrong = _draw(rng, WRONG_STATES)
  if start_state is None:
    start_state = {"scatter": ALL_ON_TABLE, "partial": PARTIAL_STATE, "wrong": drawn_wrong}[base_bucket]
  if not is_valid(start_state) or start_state == GOAL_STATE:
    raise ValueError(f"bad start state {start_state!r}")
  if stratum is None:
    stratum = drawn_stratum if start_state == ALL_ON_TABLE else "default"
  move = next_move(start_state)
  mover, target = move
  end_state = apply_move(start_state, move)

  # Positions: one table spot per tower; stacked cubes within 3 mm of the cube below.
  tws = towers(start_state)
  bases = _place_towers(rng, tws, stratum if len(tws) == 3 else "default")
  cubes: dict[str, dict] = {}
  for tw, base in zip(tws, bases):
    xy = np.asarray(base, float)
    for level, c in enumerate(tw):
      if level:
        xy = xy + rng.uniform(-CUBE_ON_CUBE_XY, CUBE_ON_CUBE_XY, 2)
      cubes[c] = {"size": SIZES[c], "xy": [float(xy[0]), float(xy[1])],
                  "z": centre_height(start_state, c) + START_GAP * (level + 1),
                  "yaw": float(rng.uniform(-math.pi, math.pi)),
                  "mass": float(rng.uniform(*MASS_RANGE[c])),
                  "friction": float(rng.uniform(0.4, 1.0))}
  for c in CUBES:
    cubes[c]["rgb"] = _cube_rgb(c, rng)
  # Where an unstacked cube goes: a free table spot in the zone, >= 20 mm from every other
  # footprint (the cubes that stay where they are).
  spot = None
  if target == TABLE:
    others = [(np.asarray(cubes[t[0]]["xy"]), _base_radius([x for x in t if x != mover]))
              for t in towers(end_state) if mover not in t]
    spot = free_spot(rng, mover, others)
  noise = None
  if bucket == "noise":
    noise = {**NOISE, "seed": int(rng.integers(2 ** 31))}

  others = [c for c in CUBES if c != mover]
  m = cubes[mover]
  v.obj_xy, v.obj_yaw, v.obj_size = m["xy"], m["yaw"], m["size"]
  v.obj_mass, v.obj_friction, v.object_rgb = m["mass"], m["friction"], m["rgb"]
  bodies = [{"name": f"extra_{k}", "shape": "box", "size": cubes[c]["size"],
             "height": cubes[c]["size"], "mass": cubes[c]["mass"], "friction": cubes[c]["friction"],
             "rgb": cubes[c]["rgb"], "xy": cubes[c]["xy"], "yaw": cubes[c]["yaw"], "z": cubes[c]["z"]}
            for k, c in enumerate(others)]
  gaps = {}
  for a, b in itertools.combinations(range(len(tws)), 2):
    d = float(np.linalg.norm(np.asarray(bases[a]) - np.asarray(bases[b])))
    gaps["".join(sorted(tws[a][0] + tws[b][0]))] = round((d - _base_radius(tws[a]) - _base_radius(tws[b])) * 1e3, 1)
  v.extras = {**v.extras, "free_bodies": bodies, "object_z": m["z"], "no_bin": True,
              "stack": {"start_state": start_state, "start_desc": describe(start_state),
                        "move": list(move), "moved_cube": mover, "end_state": end_state,
                        "bucket": bucket, "base_bucket": base_bucket,
                        "stratum": stratum if start_state == ALL_ON_TABLE else None,
                        "free_spot": None if spot is None else [float(spot[0]), float(spot[1])],
                        "gaps_mm": gaps, "zone": {"r": list(ZONE_R), "az_deg": ZONE_AZ_DEG},
                        "bodies": {"object": mover, **{b["name"]: c for b, c in zip(bodies, others)}},
                        "sizes": dict(SIZES), "instruction": INSTRUCTION, "action_noise": noise}}
  return v


def variation_for_move(v: Variation, state: str, move: tuple[str, str], spot=None) -> Variation:
  """A copy of ``v`` describing another move in the SAME physical scene (tower runs): object
  size = the cube that moves now. Only the planner reads it; the scene is not rebuilt."""
  w = copy.deepcopy(v)
  cube = move[0]
  w.obj_size = SIZES[cube]
  w.extras["stack"].update({"start_state": state, "move": list(move), "moved_cube": cube,
                            "end_state": apply_move(state, move),
                            "free_spot": None if spot is None else [float(spot[0]), float(spot[1])]})
  return w
