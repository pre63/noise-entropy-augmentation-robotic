import os

import gymnasium as gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sb3.trpo import TRPO


class RandomizationWrapper(gym.Wrapper):
  """
    An extended domain randomization wrapper for MuJoCo environments in Gymnasium.
    This wrapper randomizes several parameters (body masses, joint damping, geom friction, and gravity)
    by multiplying them with random factors within specified ranges each time the environment is reset.

    Presets for difficulty levels based on common practices in domain randomization research:
    - 'easy': Small variations for mild randomization (e.g., masses 0.9-1.1x).
    - 'hard': Moderate variations (e.g., masses 0.7-1.3x).
    - 'hardcore': Extreme variations (e.g., masses 0.5-1.5x).

    Ranges are relative to original values and inspired by literature such as:
    - Tobin et al. (Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World)
    - Papers on sim-to-real transfer for MuJoCo tasks (e.g., varying masses by 50-150%, friction 50-200%, damping similarly, gravity mildly).

    You can customize by passing a dict of ranges instead of a preset string.

    Example usage:
    env = gym.make("HalfCheetah-v5")
    env = RandomizationWrapper(env, difficulty='hard')
    # Or custom: env = RandomizationWrapper(env, custom_ranges={'masses': (0.8, 1.2), 'damping': (0.7, 1.3)})
    """

  PRESETS = {
    "easy": {"masses": (0.9, 1.1), "damping": (0.95, 1.05), "friction": (0.95, 1.05), "gravity": (0.98, 1.02)},  # Mild gravity variation
    "hard": {"masses": (0.7, 1.3), "damping": (0.8, 1.2), "friction": (0.8, 1.2), "gravity": (0.95, 1.05)},
    "hardcore": {"masses": (0.5, 1.5), "damping": (0.5, 1.5), "friction": (0.5, 2.0), "gravity": (0.9, 1.1)},
  }

  def __init__(self, env, difficulty="hard", custom_ranges=None):
    super().__init__(env)
    if not hasattr(self.unwrapped, "model"):
      raise ValueError("This wrapper is designed for MuJoCo-based environments.")

    # Store original parameters
    self.original_masses = self.unwrapped.model.body_mass.copy()
    self.original_damping = self.unwrapped.model.dof_damping.copy()
    self.original_friction = self.unwrapped.model.geom_friction.copy()
    self.original_gravity = self.unwrapped.model.opt.gravity.copy()

    if custom_ranges:
      self.ranges = custom_ranges
    elif difficulty in self.PRESETS:
      self.ranges = self.PRESETS[difficulty]
    else:
      raise ValueError(f"Invalid difficulty: {difficulty}. Choose from {list(self.PRESETS.keys())} or provide custom_ranges.")

    # Validate ranges
    expected_keys = {"masses", "damping", "friction", "gravity"}
    if not all(k in expected_keys for k in self.ranges):
      raise ValueError(f"Ranges must include keys: {expected_keys}")

  def reset(self, **kwargs):
    # Sample random factors for each parameter
    if "masses" in self.ranges:
      mass_factors = np.random.uniform(*self.ranges["masses"], size=self.original_masses.shape)
      self.unwrapped.model.body_mass[:] = self.original_masses * mass_factors

    if "damping" in self.ranges:
      damping_factors = np.random.uniform(*self.ranges["damping"], size=self.original_damping.shape)
      self.unwrapped.model.dof_damping[:] = self.original_damping * damping_factors

    if "friction" in self.ranges:
      # Typically randomize sliding friction (first column)
      friction_factors = np.random.uniform(*self.ranges["friction"], size=self.original_friction.shape[0])
      self.unwrapped.model.geom_friction[:, 0] = self.original_friction[:, 0] * friction_factors

    if "gravity" in self.ranges:
      gravity_factor = np.random.uniform(*self.ranges["gravity"])
      self.unwrapped.model.opt.gravity[2] = self.original_gravity[2] * gravity_factor  # Z-component

    # Call the original reset
    return super().reset(**kwargs)


def main():
  # Paths for saved model and vecnormalize
  MODEL_PATH = "trpo_halfcheetah.zip"
  VECNORM_PATH = "vecnormalize.pkl"

  # Function to create the base environment
  def make_env():
    return gym.make("HalfCheetah-v5")

  # Load or train the model
  if not os.path.exists(MODEL_PATH) or not os.path.exists(VECNORM_PATH):
    print("Training new model...")
    # Create vectorized environment with n_envs=2 as per RL-Zoo
    train_env = DummyVecEnv([make_env for _ in range(2)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Initialize TRPO with RL-Zoo hyperparameters
    model = TRPO(
      policy="MlpPolicy",
      env=train_env,
      batch_size=128,
      cg_damping=0.1,
      cg_max_steps=25,
      gae_lambda=0.95,
      gamma=0.99,
      learning_rate=0.001,
      n_critic_updates=20,
      n_steps=1024,
      target_kl=0.04,
      verbose=1,
    )

    # Train for 1M timesteps
    model.learn(total_timesteps=1000000)

    # Save model and vecnormalize
    model.save(MODEL_PATH)
    train_env.save(VECNORM_PATH)
    print("Model trained and saved.")
  else:
    print("Loading existing model...")
    # Load vecnormalize and model
    dummy_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(VECNORM_PATH, dummy_env)
    model = TRPO.load(MODEL_PATH, env=vec_env)

  # Now evaluate on standard and all presets
  presets = [None, "easy", "hard", "hardcore"]  # None for standard (no randomization)

  for preset in presets:
    if preset is None:
      print("\nEvaluating on standard HalfCheetah-v5...")
      eval_env_base = gym.make("HalfCheetah-v5")
    else:
      print(f"\nEvaluating on {preset} preset...")
      eval_env_base = RandomizationWrapper(gym.make("HalfCheetah-v5"), difficulty=preset)

    # Vectorize eval env
    eval_vec_env = DummyVecEnv([lambda: eval_env_base])

    # Apply VecNormalize in evaluation mode (no training)
    eval_vec_env = VecNormalize(eval_vec_env, training=False, norm_obs=True, norm_reward=False)

    # Copy running statistics from training vecnormalize for fair evaluation
    eval_vec_env.obs_rms = vec_env.obs_rms.copy()
    eval_vec_env.ret_rms = vec_env.ret_rms  # Though norm_reward=False, it's None

    # Evaluate policy
    mean_reward, std_reward = evaluate_policy(model, eval_vec_env, n_eval_episodes=100, deterministic=True)

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
  main()
