import copy
import datetime
import errno
import fcntl
import itertools
import os
import pickle
import random
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from sb3_contrib.common.utils import conjugate_gradient_solver
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from sb3.noise import MonitoredEntropyInjectionWrapper
from sb3.trpo import TRPO
from sb3.trpor import TRPOR
from scripts.experiments import RewardLoggerCallback, load_hyperparams, make_env, make_video_env, record_best_model_video


def model_name(config: Dict) -> str:
  model_class = config["model_class"]
  name = "" + model_class.__name__
  for k, v in config.items():
    if k not in ["env_id", "model_class", "timesteps"]:
      if k == "entropy_mode":
        v = "BONUS" if v == "bonus" else "PENALTY"
      elif k == "normalize_entropy":
        v = "NORM" if v else "NONORM"
      elif k == "target_kl":
        v = f"{v}KL"
      elif k == "n_critic_updates":
        v = f"{v}CU"
      elif k == "cg_max_steps":
        v = f"{v}CG"
      elif k == "learning_rate":
        v = f"{v}LR"
      elif k == "ent_coef":
        v = f"{v}EC"
      else:
        v = f"{v}{k.upper()}"
      name += f"_{v}"
  return name


class CheckpointWithDataCallback(CheckpointCallback):
  def __init__(self, *args, reward_logger: RewardLoggerCallback, pkl_path: str, total_timesteps: int, **kwargs):
    super().__init__(*args, **kwargs)
    self.reward_logger = reward_logger
    self.pkl_path = pkl_path
    self.total_timesteps = total_timesteps

  def _on_step(self) -> bool:
    if self.n_calls % self.save_freq == 0:
      self.model.save(self._checkpoint_path())
      self.save_data()
    return True

  def save_data(self):
    raw_step_rewards = self.reward_logger.step_rewards
    if len(raw_step_rewards) > self.total_timesteps:
      raw_step_rewards = raw_step_rewards[: self.total_timesteps]
    episode_infos = self.reward_logger.episode_infos
    run_data = {
      "timesteps": list(range(1, len(raw_step_rewards) + 1)),
      "step_rewards": raw_step_rewards,
      "episode_rewards": [ep["reward"] for ep in episode_infos],
      "episode_end_timesteps": [ep["end_timestep"] for ep in episode_infos],
      "inference_mean_reward": 0.0,  # Placeholder
      "inference_std_reward": 0.0,
      "clean_inference_mean_reward": 0.0,
      "clean_inference_std_reward": 0.0,
      "rollout_metrics": self.model.rollout_metrics,
    }
    with open(self.pkl_path, "wb") as f:
      pickle.dump(run_data, f)


def find_latest_checkpoint(compare_dir: str, key: str) -> Tuple[Optional[str], int]:
  checkpoints = [f for f in os.listdir(compare_dir) if f.startswith(f"{key}_") and f.endswith("_steps.zip")]
  if not checkpoints:
    return None, 0
  steps = [int(f.split("_")[-2]) for f in checkpoints]
  max_step = max(steps)
  latest = [f for f in checkpoints if int(f.split("_")[-2]) == max_step][0]
  return os.path.join(compare_dir, latest), max_step


def generate_experiments():
  """
    Generate the list of experiment configurations.

    Returns:
        list: A list of dictionaries, each representing an experiment config.
    """
  # Define environments and their total timesteps
  env_timesteps = {
    "HalfCheetah-v5": 2_000_000,
    "Humanoid-v5": 20_000_000,
    "HumanoidStandup-v5": 20_000_000,
    "Swimmer-v5": 50_000,
    "Hopper-v5": 1_000_000,
    "Walker2d-v5": 2_000_000,
  }

  # Define TRPO configurations
  trpo_configs = [
    {"cg_max_steps": 25, "target_kl": 0.01, "n_critic_updates": 20, "learning_rate": 0.001},
    {"cg_max_steps": 5, "target_kl": 0.3, "n_critic_updates": 5, "learning_rate": 0.0015},
  ]

  # Define TRPOR configurations
  trpor_configs = [
    {"entropy_mode": "bonus", "ent_coef": 0.0001, "normalize_entropy": False},
    {"entropy_mode": "bonus", "ent_coef": 0.0001, "normalize_entropy": True},
    {"entropy_mode": "penalty", "ent_coef": 0.05, "normalize_entropy": False},
    {"entropy_mode": "penalty", "ent_coef": 0.05, "normalize_entropy": True},
  ]

  experiments = []

  for env_id, timesteps in env_timesteps.items():
    # Add TRPO experiments
    for params in trpo_configs:
      config = {
        "timesteps": timesteps,
        "env_id": env_id,
        "model_class": TRPO,
      }
      config.update(params)
      experiments.append(config)

    # Add TRPOR experiments
    for params in trpor_configs:
      config = {
        "timesteps": timesteps,
        "env_id": env_id,
        "model_class": TRPOR,
      }
      config.update(params)
      experiments.append(config)

  return experiments


if __name__ == "__main__":
  # not relevant for research concerns
  n_envs = 8
  n_eval_episodes = 100

  # Experimentation schedule
  run_indicies = list(range(0, 19))

  # Per-environment overrides
  # Add any environment-specific hyperparameter overrides here
  # These will be applied after loading default hyperparams and config-specific params
  per_env_overrides: Dict[str, Dict[str, any]] = {
    # Example:
    # "Humanoid-v5": {"learning_rate": 0.0005},
    # "HalfCheetah-v5": {"target_kl": 0.02},
  }

  all_configs = generate_experiments()

  for run in run_indicies:
    for config in all_configs:
      env_id = config["env_id"]
      model_class = config["model_class"]
      total_timesteps = config["timesteps"]

      key = model_name(config) + f"_run{run}"
      compare_dir_root = "bp_comparison"
      compare_dir = os.path.join(compare_dir_root, f"{env_id}")
      os.makedirs(compare_dir, exist_ok=True)

      pkl_path = os.path.join(compare_dir, f"{key}.pkl")
      final_zip_path = os.path.join(compare_dir, f"{key}.zip")

      env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])
      hyperparams = load_hyperparams(model_class, env_id)
      if hyperparams is None:
        hyperparams = {}

      hyperparams.update({k: v for k, v in config.items() if k not in ["env_id", "model_class", "timesteps"]})
      hyperparams.update(per_env_overrides.get(env_id, {}))

      save_freq = max(1, total_timesteps // 20)  # Save approximately 20 times during training

      reward_logger = RewardLoggerCallback()

      current_timesteps = 0
      rollout_metrics = {}
      step_rewards = []
      episode_infos = []
      model = None

      if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
          run_data = pickle.load(f)
        current_timesteps = len(run_data["step_rewards"])
        if current_timesteps >= total_timesteps:
          print(f"Skipping existing complete run {env_id}/{key}")
          env.close()
          continue
        else:
          print(f"Continuing incomplete run {env_id}/{key} from {current_timesteps} to {total_timesteps}")
          latest_zip, loaded_steps = find_latest_checkpoint(compare_dir, key)
          if latest_zip and loaded_steps == current_timesteps:
            model = model_class.load(latest_zip, env=env)
            rollout_metrics = run_data["rollout_metrics"]
            step_rewards = run_data["step_rewards"]
            episode_infos = run_data["episode_infos"]
          else:
            print(f"No matching checkpoint found or mismatch in steps. Starting from scratch.")
            current_timesteps = 0

      if model is None:
        print(f"Starting new run {env_id}/{key}")
        model = model_class("MlpPolicy", env, **hyperparams, verbose=0)

      remaining = total_timesteps - current_timesteps

      checkpoint_callback = CheckpointWithDataCallback(
        save_freq=save_freq,
        save_path=compare_dir,
        name_prefix=key,
        reward_logger=reward_logger,
        pkl_path=pkl_path,
        total_timesteps=total_timesteps,
        save_vecnormalize=False,
        save_replay_buffer=False,
        verbose=1,
      )

      model.learn(total_timesteps=remaining, callback=[checkpoint_callback, reward_logger])

      # Append new data for continuation
      step_rewards += reward_logger.step_rewards
      for ep in reward_logger.episode_infos:
        ep["end_timestep"] += current_timesteps
        episode_infos.append(ep)

      for k in model.rollout_metrics:
        rollout_metrics[k] = rollout_metrics.get(k, []) + model.rollout_metrics.get(k, [])

      if len(step_rewards) > total_timesteps:
        step_rewards = step_rewards[:total_timesteps]

      model.save(final_zip_path)

      mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
      env.close()

      # Evaluate inference stability on no-noise env
      clean_env = SubprocVecEnv([make_env(env_id, None) for _ in range(n_envs)])
      clean_mean_reward, clean_std_reward = evaluate_policy(model, clean_env, n_eval_episodes=n_eval_episodes, deterministic=True)
      clean_env.close()

      run_data = {
        "timesteps": list(range(1, len(step_rewards) + 1)),
        "step_rewards": step_rewards,
        "episode_rewards": [ep["reward"] for ep in episode_infos],
        "episode_end_timesteps": [ep["end_timestep"] for ep in episode_infos],
        "inference_mean_reward": float(mean_reward),
        "inference_std_reward": float(std_reward),
        "clean_inference_mean_reward": float(clean_mean_reward),
        "clean_inference_std_reward": float(clean_std_reward),
        "rollout_metrics": rollout_metrics,
      }

      with open(pkl_path, "wb") as f:
        pickle.dump(run_data, f)

  # After all runs, generate and print the hypothesis report
  print(f"\nGenerating hypothesis report... {compare_dir_root}")

  # Collect data for bonus and penalty modes with norm/nonorm split
  data = {
    "bonus": {
      "nonorm": {"reg_ratios": [], "raw_improvs": [], "ls_success": [], "ls_coeffs": [], "final_rewards": []},
      "norm": {"reg_ratios": [], "raw_improvs": [], "ls_success": [], "ls_coeffs": [], "final_rewards": []},
    },
    "penalty": {
      "nonorm": {"reg_ratios": [], "raw_improvs": [], "ls_success": [], "ls_coeffs": [], "final_rewards": []},
      "norm": {"reg_ratios": [], "raw_improvs": [], "ls_success": [], "ls_coeffs": [], "final_rewards": []},
    },
  }
  # compare_dir_root walthe dirs per env
  for env in os.listdir(compare_dir_root):
    compare_dir = os.path.join(compare_dir_root, env)
    pkl_files = [f for f in os.listdir(compare_dir) if f.endswith(".pkl")]

    for f in pkl_files:
      # Parse filename to get config
      parts = f.replace(".pkl", "").split("_")
      ent_coef_str = 0.0
      entropy_mode = None
      norm_str = None
      run_str = 0.0
      for part in parts:
        if part.startswith("EC"):
          ent_coef_str = part.replace("EC", "")
        elif part in ["BONUS", "PENALTY"]:
          entropy_mode = "bonus" if part == "BONUS" else "penalty"
        elif part in ["NORM", "NONORM"]:
          norm_str = "norm" if part == "NORM" else "nonorm"
        elif part.startswith("run"):
          run_str = part

      if entropy_mode is None or norm_str is None:
        continue

      try:
        ent_coef = float(ent_coef_str)
        run = int(run_str.replace("run", ""))
      except ValueError:
        continue  # Skip invalid parses

      pkl_path = os.path.join(compare_dir, f)
      with open(pkl_path, "rb") as ff:
        run_data = pickle.load(ff)

      # Extraction: rollout_metrics is dict of lists (per-update means)
      rollout_metrics = run_data["rollout_metrics"]

      # Get mean over updates for the run
      reg_ratios = rollout_metrics.get("reg_ratio_mean", [])
      raw_improvs = rollout_metrics.get("raw_improv_mean", [])
      ls_success = rollout_metrics.get("line_search_success_mean", [])
      ls_coeffs = rollout_metrics.get("ls_coeff_mean", [])

      mode_data = data[entropy_mode][norm_str]
      mode_data["reg_ratios"].append(np.mean(reg_ratios) if reg_ratios else 0)
      mode_data["raw_improvs"].append(np.mean(raw_improvs) if raw_improvs else 0)
      mode_data["ls_success"].append(np.mean(ls_success) if ls_success else 0)
      mode_data["ls_coeffs"].append(np.mean(ls_coeffs) if ls_coeffs else 0)

      mode_data["final_rewards"].append(run_data["inference_mean_reward"])

    # Compute aggregates (mean ± std across runs)
    def compute_agg(data_dict, key):
      values = data_dict[key]
      if values:
        return f"{np.mean(values):.4f} ± {np.std(values):.4f}"
      return "N/A"

    # Print report
    print("\nHypothesis Report: Bonus vs Penalty Mode (Split by Norm/Nonorm)")
    print("==================================================================")
    print("Note: Bonus uses ent_coef=0.0001, Penalty uses ent_coef=0.05")
    print("Hypothesis 1: Bonus mode leads to entropy hijacking (high reg_ratio, low raw_improvement, high ls_coeff ~1)")
    print("Hypothesis 2: Penalty mode avoids hijacking (low reg_ratio, high raw_improvement, lower ls_coeff, better performance)")
    print("\nMetric                  | Bonus-Nonorm        | Bonus-Norm          | Penalty-Nonorm      | Penalty-Norm")
    print("------------------------|---------------------|---------------------|---------------------|-------------")
    print(
      f"Reg Ratio (mean±std)    | {compute_agg(data['bonus']['nonorm'], 'reg_ratios')} | {compute_agg(data['bonus']['norm'], 'reg_ratios')} | {compute_agg(data['penalty']['nonorm'], 'reg_ratios')} | {compute_agg(data['penalty']['norm'], 'reg_ratios')}"
    )
    print(
      f"Raw Improvement         | {compute_agg(data['bonus']['nonorm'], 'raw_improvs')} | {compute_agg(data['bonus']['norm'], 'raw_improvs')} | {compute_agg(data['penalty']['nonorm'], 'raw_improvs')} | {compute_agg(data['penalty']['norm'], 'raw_improvs')}"
    )
    print(
      f"LS Success Rate         | {compute_agg(data['bonus']['nonorm'], 'ls_success')} | {compute_agg(data['bonus']['norm'], 'ls_success')} | {compute_agg(data['penalty']['nonorm'], 'ls_success')} | {compute_agg(data['penalty']['norm'], 'ls_success')}"
    )
    print(
      f"Avg LS Coeff            | {compute_agg(data['bonus']['nonorm'], 'ls_coeffs')} | {compute_agg(data['bonus']['norm'], 'ls_coeffs')} | {compute_agg(data['penalty']['nonorm'], 'ls_coeffs')} | {compute_agg(data['penalty']['norm'], 'ls_coeffs')}"
    )
    print(
      f"Final Mean Reward       | {compute_agg(data['bonus']['nonorm'], 'final_rewards')} | {compute_agg(data['bonus']['norm'], 'final_rewards')} | {compute_agg(data['penalty']['nonorm'], 'final_rewards')} | {compute_agg(data['penalty']['norm'], 'final_rewards')}"
    )
    print("==================================================================")
    # print how many runs were aggregated
    print(
      f"Runs Aggregated         | {len(data['bonus']['nonorm']['final_rewards'])}                 | {len(data['bonus']['norm']['final_rewards'])}                 | {len(data['penalty']['nonorm']['final_rewards'])}                 | {len(data['penalty']['norm']['final_rewards'])}"
    )
