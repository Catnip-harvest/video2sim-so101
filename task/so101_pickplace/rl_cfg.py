"""PPO settings (rsl-rl via mjlab), adapted from mjlab's YAM lift-cube runner cfg.

10 Hz control, 15 s episodes -> 150 steps; 32 steps/env/iteration. Untuned:
the numbers are a starting point for the MI300X run, not a result.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def so101_pickplace_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.6,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.98,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="so101_pickplace_state",
    logger="tensorboard",  # no account needed on the training box
    upload_model=False,
    save_interval=100,
    num_steps_per_env=32,
    max_iterations=4_000,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
  )
