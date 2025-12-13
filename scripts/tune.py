import os

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from sb3.trpor import TRPOR
from scripts.experiments import make_env


def load_hyperparams(model_class, env_id):
  model_name = model_class.__name__.lower()
  yaml_path = f"hyperparameters/{model_name}.yaml"
  if not os.path.exists(yaml_path):
    print(f"Warning: Hyperparams file {yaml_path} not found. Using defaults.")
    return {}
  with open(yaml_path, "r") as f:
    hyperparams = yaml.safe_load(f)
  return hyperparams.get(env_id, {})


def train_and_evaluate(env_id, ent_coef, n_envs, total_timesteps, n_eval_episodes, noise_level=None):
  if noise_level is None:
    env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])
  else:
    noise_configs = [
      {
        "noise_type": "uniform",
        "noise_level": noise_level,
        "component": "action",
      }
    ]
    env = SubprocVecEnv([make_env(env_id, noise_configs) for _ in range(n_envs)])
  hyperparams = load_hyperparams(TRPOR, env_id)
  hyperparams["ent_coef"] = ent_coef
  model = TRPOR("MlpPolicy", env, **hyperparams, verbose=0)
  model.learn(total_timesteps=total_timesteps)
  mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
  env.close()
  return mean_reward, std_reward


# def print_report(results):
#   # Print optimal ent_coef for each env and noise level
#   print("\nOptimal ent_coef for each environment and noise level:")
#   for noise_level in sorted(results.keys()):
#     print(f"\nNoise level: {noise_level}")
#     for env_id in sorted(results[noise_level].keys()):
#       best_ent_coef, best_mean, best_std = results[noise_level][env_id]
#       print(f"  {env_id}: {best_ent_coef} (mean: {best_mean:.2f}, std: {best_std:.2f})")


# Define print_report to group by env, compute mean/std of means, sort by mean desc
def print_report(results):
  for env_id in results:
    print(f"\nResults for {env_id}:")
    config_stats = []
    for key, data in results[env_id].items():
      if len(data["means"]) > 0:
        mean_of_means = np.mean(data["means"])
        std_of_means = np.std(data["means"]) if len(data["means"]) > 1 else 0.0
        num_runs = len(data["means"])
        config_stats.append((key, mean_of_means, std_of_means, num_runs))
    sorted_stats = sorted(config_stats, key=lambda x: x[1], reverse=True)
    print("Noise | Ent_Coef | Mean | Std | Num_Runs")
    for (noise, coef), m, s, n in sorted_stats:
      noise_str = "None" if noise is None else noise
      print(f"{noise_str} | {coef} | {m:.2f} | {s:.2f} | {n}")


envs = ["HalfCheetah-v5", "Hopper-v5", "Swimmer-v5", "Humanoid-v5", "HumanoidStandup-v5"]
ent_coefs = sorted(
  set(
    [
      0.00001,
      0.00005,
      0.0001,
      0.0002,
      0.0003,
      0.0004,
      0.0005,
      0.0006,
      0.0007,
      0.0008,
      0.0009,
      0.001,
      0.002,
      0.003,
      0.004,
      0.005,
      0.006,
      0.007,
      0.008,
      0.009,
      0.01,
      0.02,
      0.03,
      0.04,
      0.05,
      0.06,
      0.07,
      0.08,
      0.09,
      0.1,
      0.2,
      0.3,
      0.4,
      0.5,
    ]
  )
)

if __name__ == "__main__":
  dry_run = False  # Set to True for quick testing

  n_envs = 12
  total_timesteps = 2 if dry_run else 1_000_000
  n_eval_episodes = 1 if dry_run else 20

  results = {}  # env -> {(noise, ent_coef): {"means": [], "stds": []}}

  # Pre-selected top 4 ent_coefs for HalfCheetah, Hopper, Swimmer at noise=0.1
  selected_coefs = {
    "HalfCheetah-v5": [0.0002, 0.005, 0.00005, 0.0003],
    "Hopper-v5": [0.007, 0.0005, 0.009, 0.0003],
    "Swimmer-v5": [0.009, 0.007, 0.006, 0.03],
  }

  # if dry_run: use only one ent_coef per env for quick testing
  if dry_run:
    for env_id in selected_coefs:
      selected_coefs[env_id] = [selected_coefs[env_id][0]]

  # For HalfCheetah, Hopper, Swimmer: 10 runs each for the 4 selected ent_coefs at noise=0.1
  for env_id in ["HalfCheetah-v5", "Hopper-v5", "Swimmer-v5"]:
    if env_id not in results:
      results[env_id] = {}
    noise_level = 0.1
    for ent_coef in selected_coefs[env_id]:
      key = (noise_level, ent_coef)
      results[env_id][key] = {"means": [], "stds": []}
      for run in range(10):
        print(f"Evaluating {env_id} with ent_coef={ent_coef} and noise={noise_level} (run {run+1}/10)")
        mean_reward, std_reward = train_and_evaluate(env_id, ent_coef, n_envs, total_timesteps, n_eval_episodes, noise_level)
        results[env_id][key]["means"].append(mean_reward)
        results[env_id][key]["stds"].append(std_reward)
        # Print result as we go
        print(f"  Result for run {run+1}: mean: {mean_reward:.2f}, std: {std_reward:.2f}")
        # Compute and print running average
        running_means = results[env_id][key]["means"]
        running_avg = np.mean(running_means)
        running_std_of_means = np.std(running_means) if len(running_means) > 1 else 0.0
        print(f"  Running average after {run+1} runs: mean: {running_avg:.2f}, std of means: {running_std_of_means:.2f}")

  print_report(results)

  # if sry run only one ent_coef per humanoid env for quick testing
  if dry_run:
    ent_coefs = [0.0002]

  # For Humanoid and HumanoidStandup: 1 run each for full ent_coefs at noise=0.1 and 0.2
  humanoid_envs = ["Humanoid-v5", "HumanoidStandup-v5"]
  noise_levels_humanoid = [0.1, 0.2]
  if dry_run:
    noise_levels_humanoid = [0.1]

  for env_id in humanoid_envs:
    if env_id not in results:
      results[env_id] = {}
    for noise_level in noise_levels_humanoid:
      for ent_coef in ent_coefs:
        key = (noise_level, ent_coef)
        results[env_id][key] = {"means": [], "stds": []}
        print(f"Evaluating {env_id} with ent_coef={ent_coef} and noise={noise_level} (initial run)")
        mean_reward, std_reward = train_and_evaluate(env_id, ent_coef, n_envs, total_timesteps, n_eval_episodes, noise_level)
        results[env_id][key]["means"].append(mean_reward)
        results[env_id][key]["stds"].append(std_reward)
        # Print result as we go
        print(f"  Result: mean: {mean_reward:.2f}, std: {std_reward:.2f}")

  print_report(results)

  # Select top 4 configs (noise, ent_coef) per humanoid env based on initial mean, add 9 more runs each
  for env_id in humanoid_envs:
    config_means = []
    for key in results[env_id]:
      initial_mean = results[env_id][key]["means"][0]  # Since only 1 run so far
      config_means.append((key, initial_mean))
    sorted_configs = sorted(config_means, key=lambda x: x[1], reverse=True)
    top_4 = sorted_configs[:4]
    for (noise_level, ent_coef), _ in top_4:
      key = (noise_level, ent_coef)
      for run in range(1, 10):  # 9 more runs
        print(f"Evaluating {env_id} with ent_coef={ent_coef} and noise={noise_level} (additional run {run}/9)")
        mean_reward, std_reward = train_and_evaluate(env_id, ent_coef, n_envs, total_timesteps, n_eval_episodes, noise_level)
        results[env_id][key]["means"].append(mean_reward)
        results[env_id][key]["stds"].append(std_reward)
        # Print result as we go
        print(f"  Result for additional run {run}: mean: {mean_reward:.2f}, std: {std_reward:.2f}")
        # Compute and print running average
        running_means = results[env_id][key]["means"]
        running_avg = np.mean(running_means)
        running_std_of_means = np.std(running_means) if len(running_means) > 1 else 0.0
        print(f"  Running average after {run+1} total runs: mean: {running_avg:.2f}, std of means: {running_std_of_means:.2f}")

  print_report(results)

  # For Walker2d: 10 runs each for full sweep of cohefs at 0.1 then 10 for top 4
  env_id = "Walker2d-v5"
  results[env_id] = {}

  for noise_level in [0.1]:
    for ent_coef in ent_coefs:
      key = (noise_level, ent_coef)
      results[env_id][key] = {"means": [], "stds": []}
      for run in range(1):
        print(f"Evaluating {env_id} with ent_coef={ent_coef} and noise={noise_level} (run {run+1}/10)")
        mean_reward, std_reward = train_and_evaluate(env_id, ent_coef, n_envs, total_timesteps, n_eval_episodes, noise_level)
        results[env_id][key]["means"].append(mean_reward)
        results[env_id][key]["stds"].append(std_reward)
        # Print result as we go
        print(f"  Result for run {run+1}: mean: {mean_reward:.2f}, std: {std_reward:.2f}")
        # Compute and print running average
        running_means = results[env_id][key]["means"]
        running_avg = np.mean(running_means)
        running_std_of_means = np.std(running_means) if len(running_means) > 1 else 0.0
        print(f"  Running average after {run+1} runs: mean: {running_avg:.2f}, std of means: {running_std_of_means:.2f}")

  print_report(results)

  # Select top 4 configs (noise, ent_coef) for Walker2d based on initial means, add 9 more runs each
  config_means = []
  for key in results[env_id]:
    initial_mean = results[env_id][key]["means"][0]  # Since only 1 run so far
    config_means.append((key, initial_mean))
  sorted_configs = sorted(config_means, key=lambda x: x[1], reverse=True)
  top_4 = sorted_configs[:4]

  for (noise_level, ent_coef), _ in top_4:
    key = (noise_level, ent_coef)
    for run in range(1, 10):  # 9 more runs
      print(f"Evaluating {env_id} with ent_coef={ent_coef} and noise={noise_level} (additional run {run}/9)")
      mean_reward, std_reward = train_and_evaluate(env_id, ent_coef, n_envs, total_timesteps, n_eval_episodes, noise_level)
      results[env_id][key]["means"].append(mean_reward)
      results[env_id][key]["stds"].append(std_reward)
      # Print result as we go
      print(f"  Result for additional run {run}: mean: {mean_reward:.2f}, std: {std_reward:.2f}")
      # Compute and print running average
      running_means = results[env_id][key]["means"]
      running_avg = np.mean(running_means)
      running_std_of_means = np.std(running_means) if len(running_means) > 1 else 0.0
      print(f"  Running average after {run+1} total runs: mean: {running_avg:.2f}, std of means: {running_std_of_means:.2f}")

  print_report(results)
