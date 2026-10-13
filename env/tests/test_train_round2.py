"""CPU test of env/train_round2.py's processor patch (no model weights, ~20 s).

  python -m pytest -q -p no:cacheprovider env/tests/test_train_round2.py
Needs a MolmoAct2 SFT checkpoint dir (CKPT below) and the SFT dataset; skipped otherwise.
"""

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
CKPT = os.environ.get("R2_CKPT", "/work/outputs_large/train/molmoact2_sim_v1/checkpoints/001400/pretrained_model")
DATA = os.environ.get("R2_DATA", "/work/outputs_large/datagen/sugar_v10_n1000_molmoact2")

pytestmark = pytest.mark.skipif(not Path(CKPT).exists() or not Path(DATA).exists(), reason="no checkpoint/dataset")


def _overrides_like_lerobot_train(policy_cfg, stats):
  # Copied from lerobot 0.6.1 scripts/lerobot_train.py (pretrained_path set, not resume).
  pre = {"device_processor": {"device": "cpu"},
         "normalizer_processor": {"features": {**policy_cfg.input_features, **policy_cfg.output_features},
                                  "norm_map": policy_cfg.normalization_mapping, "stats": stats},
         "rename_observations_processor": {"rename_map": {}}}
  post = {"unnormalizer_processor": {"features": policy_cfg.output_features,
                                     "norm_map": policy_cfg.normalization_mapping, "stats": stats}}
  return pre, post


def test_unpatched_fails_patched_loads():
  import train_round2 as R
  from lerobot.configs.policies import PreTrainedConfig
  from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
  import lerobot.scripts.lerobot_train as T  # noqa: F401
  cfg = PreTrainedConfig.from_pretrained(CKPT)
  meta = LeRobotDatasetMetadata("local/sft", root=DATA)
  pre_o, post_o = _overrides_like_lerobot_train(cfg, meta.stats)
  orig = T.make_pre_post_processors
  with pytest.raises(KeyError):
    orig(cfg, pretrained_path=CKPT, dataset_stats=meta.stats, preprocessor_overrides=pre_o,
         postprocessor_overrides=post_o)
  T2 = R.patch_lerobot_train()
  try:
    pre, post = T2.make_pre_post_processors(cfg, pretrained_path=CKPT, dataset_stats=meta.stats,
                                            preprocessor_overrides=dict(pre_o),
                                            postprocessor_overrides=dict(post_o))
    names = [type(s).__name__ for s in pre.steps]
    assert "MolmoAct2MaskedNormalizerProcessorStep" in names
  finally:
    T2.make_pre_post_processors = orig


def test_stats_check_passes_on_sft_data():
  import train_round2 as R
  assert R.check(CKPT, DATA) == 0
