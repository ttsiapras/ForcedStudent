import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import time 

class CustomCartPole(CartPoleEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Custom termination logic
        x, x_dot, theta, theta_dot = obs
        custom_terminated = bool(
            x < -2.0 or x > 2.0 or theta < -0.45 or theta > 0.45
        )
        
        if(x<0.5 or x>0.9):
            reward -= 0.5
        return obs, reward, custom_terminated, truncated, info

    def reset(self, **kwargs):
        return super().reset(**kwargs)



# Create a vectorized environment wrapper around your custom class
def make_env():
    return CustomCartPole(render_mode=None)

def training_session(timesteps:int = 10000):
    vec_env = DummyVecEnv([make_env])

    # Initialize the model
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # Train the model
    model.learn(total_timesteps=timesteps)

    # Save the model
    model.save("ppo_custom_cartpole")
    
    return(model)

def load_model(path2model):
    # Load the model from file
    model = PPO.load(path2model)
    return model

def run_model(model,iterations:int = 100):
    env = CustomCartPole(render_mode="human")
    obs, info = env.reset()

    for _ in range(iterations):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

        x, x_dot, theta, theta_dot = obs
    
        print(f"\rX: {x}", end="", flush=True)
        time.sleep(0.05)
    env.close()

if __name__ == "__main__":
    trainNeew = input("New training session? [y,n]:")=='y'
    print(trainNeew)

    if(not trainNeew):
        model = load_model("ppo_custom_cartpole")
        run_model(model,1000)
    else:
        model = training_session(timesteps=40000)
        run_model(model,1000)