import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from so101_units import (ArmCalib, lerobot_to_molmoact2, lerobot_to_sim,  # noqa: E402
                         molmoact2_to_lerobot, sim_to_lerobot, subsample)

RNG = np.random.default_rng(0)
LEROBOT = np.column_stack([RNG.uniform(-100, 100, (500, 5)), RNG.uniform(0, 100, 500)])


@pytest.mark.parametrize("cal", [
  ArmCalib(),
  ArmCalib(sign=[1, -1, 1, -1, 1], offset_deg=[3, -2, 5, 0, 90], gripper_deg_at_0=0,
           gripper_deg_at_100=95),
])
def test_round_trip_sim(cal, tmp_path):
  back = sim_to_lerobot(lerobot_to_sim(LEROBOT, cal), cal)
  assert np.abs(back - LEROBOT).max() < 1e-9
  cal.save(tmp_path / "c.json")
  assert ArmCalib.load(tmp_path / "c.json") == cal


def test_round_trip_molmoact2():
  assert np.abs(molmoact2_to_lerobot(lerobot_to_molmoact2(LEROBOT)) - LEROBOT).max() < 1e-12


def test_molmoact2_matches_lerobot_doc():
  # backwardcomp doc: new_lift = -(old_lift - 90), new_elbow = old_elbow - 90
  old = np.array([-0.5, 189.1, 181.4, 60.6, -3.6, 1.1])  # MolmoAct2-SO100_101 card example
  new = molmoact2_to_lerobot(old)
  assert new[1] == pytest.approx(-(189.1 - 90))
  assert new[2] == pytest.approx(181.4 - 90)
  assert np.allclose(new[[0, 3, 4, 5]], old[[0, 3, 4, 5]])
  # that card example is the folded rest pose: lift near -100, elbow near +90 in the new frame
  assert -101 < new[1] < -97 and 89 < new[2] < 93


def test_defaults_are_the_documented_hypotheses():
  q = lerobot_to_sim(np.array([[0, 0, 0, 0, 90, 0]], float))[0]
  assert np.allclose(q[:5], 0)  # wrist_roll 90 real == 0 sim
  assert q[5] == pytest.approx(np.deg2rad(-10))
  assert lerobot_to_sim(np.array([0, 0, 0, 0, 0, 100.0]))[5] == pytest.approx(np.deg2rad(100))


def test_bad_inputs():
  with pytest.raises(ValueError):
    lerobot_to_sim(np.zeros(5))
  with pytest.raises(ValueError):
    ArmCalib(sign=[1, 1, 1, 1, 0.5])


def test_subsample():
  assert subsample(10, 30, 10).tolist() == [0, 3, 6, 9]
  assert subsample(4, 30, 30).tolist() == [0, 1, 2, 3]
  with pytest.raises(ValueError):
    subsample(10, 30, 7)
