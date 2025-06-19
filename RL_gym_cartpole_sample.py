import gymnasium as gym
import numpy as np

from gymnasium.envs.classic_control import CartPoleEnv

class CustomCartPole(CartPoleEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Custom termination logic — for example, end if pole angle is even tighter
        x, x_dot, theta, theta_dot = obs
        custom_terminated = bool(
            x < -2.0 or x > 2.0 or theta < -0.45 or theta > 0.45
        )

        return obs, reward, custom_terminated, truncated, info



# Create the environment
env = env = CustomCartPole(render_mode="human")

# Reset the environment (returns initial observation and info)
observation, info = env.reset(seed=42)

for _ in range(200):
    #env.render()  # shows the environment window if supported
    action = env.action_space.sample()  # choose a random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

print(truncated)
env.close()