import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Increase font sizes globally
plt.rcParams.update({"font.size": 14})


class LoggingUniformNoisyActionWrapper(gym.ActionWrapper):
  def __init__(self, env, noise_scale=0.1):
    super().__init__(env)
    self.noise_scale = noise_scale
    self.last_clean_action = None
    self.last_noisy_action = None
    self.base_range = 1.0

  def action(self, act):
    self.last_clean_action = act.copy()
    noisy_act = self._add_action_noise(act)
    self.last_noisy_action = noisy_act.copy()
    return noisy_act

  def _add_action_noise(self, action):
    noisy = action.copy()
    noise_scale = abs(self.noise_scale)
    noise = np.random.uniform(-noise_scale * self.base_range, noise_scale * self.base_range, size=action.shape)
    noisy += noise
    noisy = np.clip(noisy, self.action_space.low, self.action_space.high)
    return noisy


def make_env(env_id):
  def _init():
    env = gym.make(env_id)  # No render_mode needed since render=False
    env = LoggingUniformNoisyActionWrapper(env, noise_scale=0.1)
    return env

  return _init


def getTitle(env_id):
  titles = {
    "HalfCheetah-v5": "HalfCheetah",
    "Hopper-v5": "Hopper",
    "Swimmer-v5": "Swimmer",
    "Walker2d-v5": "Walker",
    "Humanoid-v5": "Humanoid",
    "HumanoidStandup-v5": "Standup",
  }
  return titles.get(env_id, env_id).lower()


# Function to collect actions
def collect_actions(noisy_env, num_steps, render=False):
  clean_actions = []
  noisy_actions = []
  obs, _ = noisy_env.reset()
  for _ in range(num_steps):
    clean_action = noisy_env.action_space.sample()
    clean_actions.append(clean_action.copy())
    noisy_env.step(clean_action)  # Adds noise and logs inside
    noisy_actions.append(noisy_env.last_noisy_action.copy())
    if render:
      noisy_env.render()
  return np.array(clean_actions), np.array(noisy_actions)


if __name__ == "__main__":
  import matplotlib as mpl

  mpl.rcParams.update(
    {
      "font.family": "monospace",
      "font.size": 19,
      "axes.labelsize": 19,
      "axes.titlesize": 24,
      "legend.fontsize": 24,
      "xtick.labelsize": 19,
      "ytick.labelsize": 19,
      "axes.grid": True,
      "grid.color": "gainsboro",
      "grid.alpha": 0.5,
    }
  )

  print("Generating z-score normalized signed delta heatmaps for multiple environments...")
  # Create the environments
  env_ids = ["HalfCheetah-v5", "Hopper-v5", "Swimmer-v5", "Walker2d-v5", "Humanoid-v5", "HumanoidStandup-v5"]

  # Parameters
  seed = 42
  num_runs = 1000
  num_steps = 50

  # Collect data for all environments
  mean_delta_matrices = {}
  std_deltas = {}
  for env_id in env_ids:
    diff_list = []
    for run in range(num_runs):
      noisy_env = make_env(env_id)()
      np.random.seed(seed + run)
      noisy_env.reset(seed=seed + run)
      clean_acts, noisy_acts = collect_actions(noisy_env, num_steps=num_steps, render=False)
      diff_acts = noisy_acts - clean_acts
      diff_list.append(diff_acts)
      noisy_env.close()
    all_diffs = np.stack(diff_list, axis=0)  # (num_runs, num_steps, num_dims)
    mean_diffs = np.mean(all_diffs, axis=0)  # (num_steps, num_dims)
    mean_delta_matrices[env_id] = mean_diffs.T  # (num_dims, num_steps)

    # Compute std_delta for z-score
    std_delta = np.std(all_diffs)
    std_deltas[env_id] = std_delta

  # Create one figure with subplots in two rows
  fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # Adjusted for 2 rows, 3 columns
  axs = axs.flatten()  # Flatten to iterate easily

  im = None  # To hold the last image for colorbar
  for i, env_id in enumerate(env_ids):
    matrix = mean_delta_matrices[env_id]
    std_delta = std_deltas[env_id]
    se = std_delta / np.sqrt(num_runs)
    if se == 0:
      normalized_matrix = np.zeros_like(matrix)
    else:
      normalized_matrix = matrix / se
    # Heatmap with simplified labels
    im = axs[i].imshow(normalized_matrix, aspect="auto", cmap="RdBu", vmin=-5, vmax=5)
    axs[i].set_xlabel("Timestep", fontsize=19)
    axs[i].set_ylabel("Dimension", fontsize=19)
    axs[i].set_title(getTitle(env_id), fontsize=24)
    axs[i].tick_params(labelsize=19)

  # Add a single shared colorbar on the right
  cb_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position: right, bottom, width, height
  cbar = fig.colorbar(im, cax=cb_ax)
  cbar.set_label("Z-score", fontsize=19)
  cbar.ax.tick_params(labelsize=19)

  plt.tight_layout(rect=[0, 0.05, 0.91, 1])  # Adjust layout to make room for caption and colorbar

  # Save to file
  os.makedirs("assets", exist_ok=True)
  plt.savefig("assets/multi_env_zscore_signed_delta_heatmaps.png")
