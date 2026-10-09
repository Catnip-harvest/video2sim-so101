"""lerobot-train for filtered-BC round 2: continue fine-tuning from a MolmoAct2 SFT
`pretrained_model/` dir (--policy.path) on a merged dataset.

Why a wrapper (measured on CPU, 9 Oct): lerobot 0.6.1's lerobot_train, given --policy.path,
loads the saved processors with overrides keyed "normalizer_processor" /
"unnormalizer_processor". MolmoAct2's steps are registered as "molmoact2_masked_normalizer" /
"molmoact2_masked_unnormalizer", so PolicyProcessorPipeline.from_pretrained raises
KeyError("Override keys ['normalizer_processor'] do not match any step ...") - after the
11 GB model has loaded. This wrapper drops override keys that the saved pipeline does not
have. The dropped overrides would have replaced the normaliser stats with the dataset's;
the checkpoint's own stats are kept instead, which is correct here: both are the
MolmoAct2-SO100_101 pretrained stats (use_pretrained_norm.py), and the saved ones also carry
the gripper mask that a plain dataset-stats override would lose. `--check` verifies the
equality before anything heavy runs.

  python env/train_round2.py --check INIT_PRETRAINED_MODEL DATASET     # CPU, seconds
  python env/train_round2.py --dry-run <lerobot-train args>             # parse + validate config
  python env/train_round2.py <lerobot-train args>                       # train
"""

import json
import sys
from pathlib import Path


def _step_keys(cfg_json: Path) -> set[str]:
  d = json.loads(cfg_json.read_text())
  keys = set()
  for st in d["steps"]:
    if st.get("registry_name"):
      keys.add(st["registry_name"])
    elif st.get("class"):
      keys.add(st["class"].rsplit(".", 1)[-1])
  return keys


def patch_lerobot_train():
  import lerobot.scripts.lerobot_train as T
  orig = T.make_pre_post_processors

  def make(policy_cfg, pretrained_path=None, **kw):
    if pretrained_path:
      for key, fname in (("preprocessor_overrides", "policy_preprocessor.json"),
                         ("postprocessor_overrides", "policy_postprocessor.json")):
        ov = kw.get(key)
        if ov:
          valid = _step_keys(Path(pretrained_path) / fname)
          dropped = sorted(k for k in ov if k not in valid)
          if dropped:
            print(f"[train_round2] {fname}: dropping overrides {dropped} (not in the saved "
                  f"pipeline; the checkpoint's own normaliser stats are kept)", flush=True)
          kw[key] = {k: v for k, v in ov.items() if k in valid}
    return orig(policy_cfg, pretrained_path=pretrained_path, **kw)

  T.make_pre_post_processors = make
  return T


def check(init: str, dataset: str) -> int:
  """Dataset stats (action/state quantiles) must equal the checkpoint normaliser's."""
  import numpy as np
  from safetensors.numpy import load_file
  init_p = Path(init)
  f = sorted(init_p.glob("policy_preprocessor_step_*_normalizer.safetensors"))
  if not f:
    print(f"no normaliser state in {init}")
    return 1
  z = load_file(str(f[0]))
  st = json.loads((Path(dataset) / "meta" / "stats.json").read_text())
  bad = []
  for feat in ("action", "observation.state"):
    for q in ("q01", "q99", "mean", "std"):
      a, b = np.asarray(z[f"{feat}.{q}"], float), np.asarray(st[feat][q], float)
      if a.shape != b.shape or np.max(np.abs(a - b)) > 1e-4:
        bad.append(f"{feat}.{q}")
  dg = json.loads((Path(dataset) / "meta" / "datagen.json").read_text())
  print(json.dumps({"init": init, "dataset": dataset, "stats_mismatch": bad,
                    "joint_frame": dg.get("joint_frame"),
                    "n_episodes": json.loads((Path(dataset) / "meta" / "info.json").read_text())["total_episodes"]}))
  return 1 if bad or dg.get("joint_frame") != "molmoact2" else 0


def main() -> None:
  if len(sys.argv) > 1 and sys.argv[1] == "--check":
    sys.exit(check(sys.argv[2], sys.argv[3]))
  dry = len(sys.argv) > 1 and sys.argv[1] == "--dry-run"
  if dry:
    sys.argv.pop(1)
    import lerobot.scripts.lerobot_train  # noqa: F401  (registers policy types, as a real run does)
    from lerobot.configs import parser
    from lerobot.configs.train import TrainPipelineConfig

    def show(cfg: TrainPipelineConfig):
      cfg.validate()
      p = cfg.policy
      print(json.dumps({"policy.type": p.type, "policy.pretrained_path": str(p.pretrained_path),
                        "checkpoint_path": p.checkpoint_path, "chunk_size": p.chunk_size,
                        "n_action_steps": p.n_action_steps, "cuda_graph": p.enable_inference_cuda_graph,
                        "optimizer_lr": p.optimizer_lr, "dataset.root": str(cfg.dataset.root),
                        "steps": cfg.steps, "batch_size": cfg.batch_size, "save_freq": cfg.save_freq,
                        "output_dir": str(cfg.output_dir), "resume": cfg.resume,
                        "use_policy_training_preset": cfg.use_policy_training_preset}, indent=1))
    show.__annotations__ = {"cfg": TrainPipelineConfig}  # parser.wrap reads the annotation
    parser.wrap()(show)()
    return
  T = patch_lerobot_train()
  T.main()


if __name__ == "__main__":
  main()
