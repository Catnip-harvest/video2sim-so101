"""Tower of Hanoi on the table: rules, optimal solver, scene layout and per-episode draws.

The puzzle. Three cubes, large (45 mm), medium (35 mm) and small (25 mm), sit on three
flat spots A (left), B (middle) and C (right). On a spot the cubes form a stack, and a
legal stack always has the larger cube below the smaller one. One move takes the TOP cube
of one spot and puts it on another spot: on the table if that spot is empty, otherwise on
top of a LARGER cube. The goal is all three cubes on spot C.

State. Because a legal stack is always ordered by size, a state is fully described by
which spot each cube is on. We write it as three letters for (large, medium, small):
"AAA" = everything on A (the start of the full puzzle), "CCC" = the goal, "ABA" = large
and small on A (small on top), medium alone on B. There are 3 x 3 x 3 = 27 states and all
of them are legal; 26 are not the goal.

Solver. Every move can be undone by the reverse move, so a breadth-first search outward
from the goal gives the number of moves left from every state, and the optimal next move
from a state is the move to a neighbour that is one step closer. From "AAA" that takes the
classic 2**3 - 1 = 7 moves.

Left and right are as seen in the front camera (``cam_head``), which is also the view a
person watching the dataset videos gets. That camera stands in front of the arm and looks
back at it, so image-right is +y in the robot base frame: A has the most negative y, C the
most positive.

Scene numbers (robot base frame, metres; PAN_AXIS is the shoulder-pan axis):
  spots       on an arc of radius SPOT_RADIUS around the pan axis at azimuths
              SPOT_AZIMUTH_DEG (A, B, C), so all three are equally reachable; each spot
              centre jitters inside a 1 cm disc per episode; centre-to-centre >= 9 cm is
              enforced (nominal spacing ~12 cm)
  cubes       edges 45 / 35 / 25 mm, masses 40-60 / 25-40 / 15-25 g, friction 0.4-1.0,
              faces within +-10 deg of square to the arm (the spot's radial direction, as
              a person would set them down; the expert grasps the face pair closest to
              tangential, see hanoi_expert.py), each cube's centre within 3 mm of its
              spot (or of the cube below it); colours: large blue-ish, medium green-ish, small red-ish, with a
              small hue / brightness jitter
  spot marks  flat visual-only discs (no collision), a dark ring around a light centre so
              they show on any table colour
  no bin      the tray of the pick-and-place task is not built
Everything else (servos, table, lights, front camera, expert tempo and pauses) is the
normal ``sample_variation`` draw, unchanged.

How the physics engine sees a Hanoi episode: the cube the expert must move is the
ordinary ``object`` body of the pick-and-place task (so the grasp, carry and verifier
machinery is reused unchanged), and the other two cubes are extra free bodies
(free_bodies.py) that may start stacked. Nothing is welded, attached or teleported.
"""

from __future__ import annotations

import colorsys
import copy
import itertools
import math
from collections import deque

import numpy as np

from .paths import TaskSpec
from .variations import PAN_AXIS, Variation, _in_reach, sample_variation

# ---- puzzle rules ------------------------------------------------------------------------
CUBES = ("L", "M", "S")  # large, medium, small: also the bottom-to-top order of any stack
CUBE_NAMES = {"L": "large", "M": "medium", "S": "small"}
SIZES = {"L": 0.045, "M": 0.035, "S": 0.025}  # cube edge (m), inside the 20-50 mm grasp band
SPOTS = ("A", "B", "C")
START_STATE = "AAA"
GOAL_STATE = "CCC"
INSTRUCTION = "make the next move to stack the tower on the right spot"

# ---- scene layout ------------------------------------------------------------------------
SPOT_RADIUS = 0.19  # m from the pan axis (the arm reaches highest around 0.15-0.20 m)
SPOT_AZIMUTH_DEG = {"A": -37.0, "B": 0.0, "C": 37.0}  # A at -y (left in cam_head), C at +y
SPOT_JITTER = 0.01  # each spot centre moves at most this far per episode
MIN_SPOT_SPACING = 0.09
CUBE_XY_JITTER = 0.003  # a cube's centre vs. its spot (or the cube it stands on)
CUBE_YAW_JITTER = math.radians(10.0)  # cube faces vs. the spot's radial direction
SPOT_MARK_RADIUS = (0.033, 0.027)  # dark ring outer radius, light centre radius
CUBE_RGB = {"L": (0.12, 0.25, 0.90), "M": (0.12, 0.70, 0.18), "S": (0.85, 0.12, 0.10)}
# Cube masses (kg). The pick-and-place cube's band is 15-41 g (spec 25 g x 0.61-1.65), and
# the grasp gate was validated there. Lighter cubes are a problem in MuJoCo's soft contact
# model: under the same jaw squeeze a contact sinks deeper the lighter the body (8 g wooden
# 25 mm cubes sank 2.1-2.6 mm into the jaw tips and failed the verifier's 2 mm
# penetration rule). So the masses stay in that band, scaled up for the larger cubes
# (dense toy blocks / printed PLA).
MASS_RANGE = {"L": (0.040, 0.060), "M": (0.025, 0.040), "S": (0.015, 0.025)}
START_GAP = 0.0004  # m per stack level: cubes are dropped into place from just above


# ---- states and moves ----------------------------------------------------------------------
def all_states() -> list[str]:
  return ["".join(p) for p in itertools.product(SPOTS, repeat=len(CUBES))]


def stacks(state: str) -> dict[str, list[str]]:
  """spot -> its cubes from bottom to top (largest first, by the stacking rule)."""
  out = {s: [] for s in SPOTS}
  for cube, spot in zip(CUBES, state):
    out[spot].append(cube)
  return out


def top_cube(state: str, spot: str) -> str | None:
  on_spot = stacks(state)[spot]
  return on_spot[-1] if on_spot else None


def is_larger(a: str, b: str) -> bool:
  return SIZES[a] > SIZES[b]


def legal_moves(state: str) -> list[tuple[str, str]]:
  """Every (from_spot, to_spot) that moves a top cube onto an empty spot or a larger cube."""
  moves = []
  for src in SPOTS:
    cube = top_cube(state, src)
    if cube is None:
      continue
    for dst in SPOTS:
      if dst == src:
        continue
      below = top_cube(state, dst)
      if below is None or is_larger(below, cube):
        moves.append((src, dst))
  return moves


def apply_move(state: str, move: tuple[str, str]) -> str:
  src, dst = move
  if move not in legal_moves(state):
    raise ValueError(f"illegal move {move} from state {state}")
  cube = top_cube(state, src)
  i = CUBES.index(cube)
  return state[:i] + dst + state[i + 1:]


def moved_cube(state: str, move: tuple[str, str]) -> str:
  return top_cube(state, move[0])


def _distances_to_goal(goal: str = GOAL_STATE) -> dict[str, int]:
  """Breadth-first search from the goal. Moves are reversible, so this is the number of
  moves each state needs to reach the goal."""
  dist = {goal: 0}
  queue = deque([goal])
  while queue:
    s = queue.popleft()
    for mv in legal_moves(s):
      t = apply_move(s, mv)
      if t not in dist:
        dist[t] = dist[s] + 1
        queue.append(t)
  return dist


DIST_TO_GOAL = _distances_to_goal()


def next_move(state: str) -> tuple[str, str] | None:
  """The optimal next (from_spot, to_spot), or None at the goal."""
  if state == GOAL_STATE:
    return None
  d = DIST_TO_GOAL[state]
  best = [mv for mv in legal_moves(state) if DIST_TO_GOAL[apply_move(state, mv)] == d - 1]
  return best[0]  # unique for this puzzle (tests check that), so the order does not matter


def optimal_moves_between(state: str) -> list[tuple[str, str]]:
  """The whole optimal move list from ``state`` to the goal."""
  out = []
  while state != GOAL_STATE:
    mv = next_move(state)
    out.append(mv)
    state = apply_move(state, mv)
  return out


NON_GOAL_STATES = [s for s in all_states() if s != GOAL_STATE]


# ---- geometry of a state ------------------------------------------------------------------
def centre_height(state: str, cube: str) -> float:
  """Height of a cube's centre above the table in a legal stack (m)."""
  below = [c for c in stacks(state)[state[CUBES.index(cube)]] if is_larger(c, cube)]
  return sum(SIZES[c] for c in below) + SIZES[cube] / 2


def stack_top(state: str, spot: str, without: str | None = None) -> float:
  """Height of the top face of a spot's stack (0 = empty spot), ignoring cube ``without``."""
  return sum(SIZES[c] for c in stacks(state)[spot] if c != without)


def spot_azimuth(xy) -> float:
  """Direction of a point seen from the shoulder-pan axis (rad, 0 = straight ahead)."""
  return math.atan2(float(xy[1]) - PAN_AXIS[1], float(xy[0]) - PAN_AXIS[0])


def nominal_spot_xy(spot: str) -> np.ndarray:
  a = math.radians(SPOT_AZIMUTH_DEG[spot])
  return PAN_AXIS + SPOT_RADIUS * np.array([math.cos(a), math.sin(a)])


# ---- per-episode draw ------------------------------------------------------------------------
def hanoi_info(v: Variation) -> dict | None:
  return (v.extras or {}).get("hanoi")


def _cube_rgb(cube: str, rng: np.random.Generator) -> list[float]:
  h, s, val = colorsys.rgb_to_hsv(*CUBE_RGB[cube])
  h = (h + rng.uniform(-0.02, 0.02)) % 1.0
  s = float(np.clip(s * rng.uniform(0.9, 1.0), 0.0, 1.0))
  val = float(np.clip(val * rng.uniform(0.85, 1.0), 0.0, 1.0))
  return [float(x) for x in colorsys.hsv_to_rgb(h, s, val)]


def _draw_spots(rng: np.random.Generator) -> dict[str, list[float]]:
  for _ in range(1000):
    xy = {}
    for spot in SPOTS:
      r = SPOT_JITTER * math.sqrt(rng.uniform())  # uniform in a 1 cm disc
      a = rng.uniform(-math.pi, math.pi)
      xy[spot] = nominal_spot_xy(spot) + r * np.array([math.cos(a), math.sin(a)])
    far_enough = all(np.linalg.norm(xy[a] - xy[b]) >= MIN_SPOT_SPACING
                     for a, b in itertools.combinations(SPOTS, 2))
    if far_enough and all(_in_reach(p) for p in xy.values()):
      return {s: [float(p[0]), float(p[1])] for s, p in xy.items()}
  raise RuntimeError("could not place the three spots")


def sample_hanoi_variation(spec: TaskSpec, base_seed: int, attempt: int, variety: bool = True,
                           start_state: str | None = None) -> Variation:
  """One single-move Hanoi episode: the normal Variation (servos, look, camera, expert
  style) re-laid-out as the Hanoi scene from a separate random stream ([seed, 13]).

  ``start_state`` None draws it uniformly from the 26 non-goal states. The expert's job
  in this episode is the solver's next move from it; the expected result is stored with
  it, so the dataset and the verifier never have to recompute it."""
  v = sample_variation(spec, base_seed, attempt, variety=variety)
  rng = np.random.default_rng([v.seed, 13])
  spots = _draw_spots(rng)
  drawn = NON_GOAL_STATES[int(rng.integers(len(NON_GOAL_STATES)))]
  state = drawn if start_state is None else start_state
  if state not in all_states():
    raise ValueError(f"unknown state {state!r}")
  move = next_move(state)
  if move is None:
    raise ValueError("the start state is already the goal: there is no next move")
  mover = moved_cube(state, move)
  # Cube positions: each cube within CUBE_XY_JITTER of the cube below it (or its spot).
  # Each cube starts START_GAP per stack level above its resting height and settles
  # (unrecorded) before the episode starts.
  cubes = {}
  for spot, on_spot in stacks(state).items():
    xy = np.asarray(spots[spot], float)
    for level, cube in enumerate(on_spot):
      xy = xy + rng.uniform(-CUBE_XY_JITTER, CUBE_XY_JITTER, 2)
      size = SIZES[cube]
      cubes[cube] = {
        "size": size, "xy": [float(xy[0]), float(xy[1])],
        "z": centre_height(state, cube) + START_GAP * (level + 1),
        "yaw": spot_azimuth(xy) + float(rng.uniform(-CUBE_YAW_JITTER, CUBE_YAW_JITTER)),
        "mass": float(rng.uniform(*MASS_RANGE[cube])),
        "friction": float(rng.uniform(0.4, 1.0)),
      }
  for cube in CUBES:  # colours drawn last, in a fixed order (render only)
    cubes[cube]["rgb"] = _cube_rgb(cube, rng)

  # The cube to move is the planner's "object"; the other two are extra free bodies.
  others = [c for c in CUBES if c != mover]
  m = cubes[mover]
  v.obj_xy, v.obj_yaw, v.obj_size = m["xy"], m["yaw"], m["size"]
  v.obj_mass, v.obj_friction, v.object_rgb = m["mass"], m["friction"], m["rgb"]
  bodies = []
  for k, cube in enumerate(others):
    c = cubes[cube]
    bodies.append({"name": f"extra_{k}", "shape": "box", "size": c["size"], "height": c["size"],
                   "mass": c["mass"], "friction": c["friction"], "rgb": c["rgb"],
                   "xy": c["xy"], "yaw": c["yaw"], "z": c["z"]})
  marks = [{"xy": spots[s], "radius": SPOT_MARK_RADIUS} for s in SPOTS]
  v.extras = {**v.extras, "free_bodies": bodies, "object_z": m["z"], "no_bin": True,
              "spot_marks": marks,
              "hanoi": {"spots": spots, "start_state": state, "move": list(move),
                        "moved_cube": mover, "end_state": apply_move(state, move),
                        "bodies": {"object": mover, **{b["name"]: c for b, c in zip(bodies, others)}},
                        "sizes": dict(SIZES), "instruction": INSTRUCTION,
                        "drawn_start_state": drawn}}
  return v


def body_of_cube(v: Variation) -> dict[str, str]:
  """cube letter -> body name in this episode's scene."""
  return {cube: body for body, cube in hanoi_info(v)["bodies"].items()}


def variation_for_move(v: Variation, positions: dict[str, np.ndarray], state: str,
                       move: tuple[str, str]) -> Variation:
  """A copy of ``v`` describing another move in the SAME physical scene (full-puzzle runs):
  object size = the cube that moves now. Only the planner reads it; the scene is not
  rebuilt."""
  w = copy.deepcopy(v)
  cube = moved_cube(state, move)
  w.obj_size = SIZES[cube]
  w.obj_xy = [float(positions[cube][0]), float(positions[cube][1])]
  h = w.extras["hanoi"]
  h.update({"start_state": state, "move": list(move), "moved_cube": cube,
            "end_state": apply_move(state, move)})
  return w
