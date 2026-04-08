"""Reinforcement Learning Trainer and Visualizer for MetaDrive Self-Driving.

Provides RL model training using Stable-Baselines3 (PPO) and interactive
animations (real 3D or 2D top-down simulation) to let the machine learn
as a self-driving traffic regulation system.

Usage:
  # Train a new model (headless, fast)
  python train_rl.py --mode train --steps 10000

  # Evaluate a trained model in full 3D simulation
  python train_rl.py --mode eval --render 3d

  # Evaluate a trained model in 2D top-down simulation
  python train_rl.py --mode eval --render 2d
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from metadrive.envs.metadrive_env import MetaDriveEnv


MODEL_PATH = "ppo_metadrive.zip"
LOG_DIR = "./rl_logs"


import gymnasium as gym
import cv2

class RewardModWrapper(gym.Wrapper):
    """Shape the reward to encourage smooth driving and punish crashes heavily."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Penalize steering heavily to keep the car going straight 
        # action[0] is steering, action[1] is acceleration/brake
        steering_penalty = (action[0] ** 2) * 0.5 
        reward -= steering_penalty
        
        # Add harsh penalties for striking objects or leaving the road
        if info.get("crash") or info.get("out_of_road") or info.get("crash_vehicle"):
            reward -= 50.0  
            
        return obs, reward, terminated, truncated, info

class SeedIgnoreWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        # SB3 passes huge/small seeds that violate MetaDrive assertions.
        # Transform them into a valid range for MetaDrive procedural envs.
        seed = kwargs.get("seed", None)
        if seed is not None:
            # Map SB3's arbitrary seed into the valid scenario range
            # Base start_seed will be 42, with 100 scenarios. 
            kwargs["seed"] = 42 + (seed % 100)
            
        # MetaDrive's reset doesn't accept 'options' currently, so we remove it
        # to prevent TypeError when Gym passes it down.
        if "options" in kwargs:
            kwargs.pop("options")
            
        return self.env.reset(**kwargs)

def make_env(render_mode="none"):
    """Create a configured MetaDrive environment for RL."""
    def _init():
        config = {
            "num_scenarios": 100,      # Focus on a smaller set to learn the basics
            "start_seed": 42,
            "traffic_density": 0.05,   # Sparse traffic to learn lane keeping first
            "horizon": 1000,           # Let it drive longer
            
            # Massive penalties for striking
            "out_of_road_penalty": 50.0,
            "crash_vehicle_penalty": 50.0,
            "crash_object_penalty": 50.0,
            "success_reward": 100.0,   # High reward for success
            
            "use_render": render_mode == "3d",
        }
        
        env = MetaDriveEnv(config)
        env = SeedIgnoreWrapper(env)
        env = RewardModWrapper(env)
        return env

    return _init


def train(total_timesteps: int):
    """Train an RL policy using PPO and save the model."""
    print("=" * 60)
    print(f"Starting RL Training for {total_timesteps} steps...")
    print("=" * 60)

    # Use a single environment for MetaDrive to avoid global Panda3D conflicts
    # Multiprocessing requires SubprocVecEnv and careful setup.
    vec_env = make_vec_env(make_env(render_mode="none"), n_envs=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path=LOG_DIR,
        name_prefix="rl_model"
    )

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=1e-4,              # Slower decay, stable convergence
        n_steps=2048,                    # More experience collected before updates
        batch_size=128,                  # Larger batch size to smooth out gradients
        n_epochs=10,                     # Pass over batch multiple times
        gamma=0.99,                      # Standard discounting
        gae_lambda=0.95,                 # Smooth advantage
        clip_range=0.2,                  # Limit PPO destruction
        ent_coef=0.005,                  # Help explore slightly instead of local maxing
        policy_kwargs=policy_kwargs      # Deeper network
    )
    
    # Check if we should load an existing model to continue training
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH} to resume training.")
        model = PPO.load(MODEL_PATH, env=vec_env)

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    print("Training complete! Saving to", MODEL_PATH)
    model.save(MODEL_PATH)
    vec_env.close()


def evaluate(render_mode: str):
    """Load a trained model and display an animation of the policy."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found. Please train first.")
        print("Run: python train_rl.py --mode train")
        return

    print("=" * 60)
    print(f"Evaluating RL Agent in {render_mode.upper()} mode...")
    print("=" * 60)

    # Initialize environment with the requested render mode
    env = make_env(render_mode)()
    model = PPO.load(MODEL_PATH, env=env)

    obs, info = env.reset()
    done = False
    episode_reward = 0.0

    while True:
        # Ask the agent for an action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        done = terminated or truncated

        # For 2D top-down simulation we manually show the rendered image
        if render_mode == "2d":
            # The MetaDrive top_down renderer returns an RGB array
            img = env.unwrapped.render(mode="top_down")
            if img is not None:
                # Convert RGB to BGR for cv2
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("MetaDrive 2D Evaluation", img_bgr)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("Exiting evaluation...")
                    break

        if done:
            print(f"Episode finished! Total reward: {episode_reward:.2f}")
            obs, info = env.reset()
            episode_reward = 0.0

    env.close()
    if render_mode == "2d":
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="MetaDrive RL Training & Visualization")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "eval"], 
        default="eval",
        help="Whether to 'train' a new model or 'eval' (watch) an existing one."
    )
    parser.add_argument(
        "--render", 
        type=str, 
        choices=["none", "2d", "3d"], 
        default="3d",
        help="Animation mode during evaluation. 2d (TopDown) or 3d (First-person)."
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=2000, 
        help="Number of timesteps to train (only used if --mode train)."
    )
    
    args = parser.parse_args()

    if args.mode == "train":
        train(args.steps)
    else:
        evaluate(args.render)


if __name__ == "__main__":
    main()
