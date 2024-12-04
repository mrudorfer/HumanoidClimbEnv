import os.path

import gymnasium as gym
import pybullet as p
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from humanoid_climb.climbing_config import ClimbingConfig


# load config, create and reset env, load policy
# config_file = './configs/config.json'
sim_config_fn = './configs/sim_config.json'
climb_config_fn = './configs/simple_train_config.json'
policy_dir = './models/f53ji5br/models/'
policies = ['best_model.zip']
stats_path = os.path.join(policy_dir, 'vecnormalize.pkl')

config = ClimbingConfig(sim_config_fn, climb_config_fn)

env_kwargs = {
    'render_mode': 'human',
    'max_ep_steps': 10_000_000,
    'config': config
}

vec_env = make_vec_env('HumanoidClimb-v0', n_envs=1, env_kwargs=env_kwargs)
vec_env = VecNormalize.load(stats_path, vec_env)
#  do not update them at test time, also don't need to normalize rewards
vec_env.training = False
vec_env.norm_reward = False
obs = vec_env.reset()

model = PPO.load(os.path.join(policy_dir, policies[0]), env=vec_env)
print(model.policy)
policy_idx = 0

# prepare variables for the while loop
done, terminated, truncated = False, False, False
score, step = 0, 0
paused = False
info = {}

print('In PyBullet window, press:')
print('\tr          reset episode')
print('\tspacebar   pause episode')
print('\tq          quit')

while True:
    if not paused:
        # use policy to predict next action, then step environment
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        # extract from vec_env
        reward, done, info = reward[0], done[0], info[0]
        score += reward
        step += 1

    if (step % 200 == 0) and not paused:
        print(f'step {step}: score {score}')

    # if episode terminates, pause (user needs to reset)
    # note that episode currently does not terminate after goal has successfully been reached
    if done and not paused:
        print(f'terminated after {step} steps. score: {score}')
        paused = True

    if step > 1 and not paused and info['stance_reached']:
        policy_idx += 1
        print(f'stance reached after {step} steps!! activating policy {policies[policy_idx][:-4]}.')
        model = PPO.load(os.path.join(policy_dir, policies[policy_idx]), env=vec_env)

    # get keys
    keys = p.getKeyboardEvents()
    # reset on r
    r_key = ord('r')
    if r_key in keys and keys[r_key] & p.KEY_WAS_TRIGGERED:
        print('resetting...')
        terminated = False
        truncated = False
        paused = False
        score = 0
        step = 0
        if policy_idx != 0:
            policy_idx = 0
            model = PPO.load(os.path.join(policy_dir, policies[policy_idx]), env=vec_env)
        obs = vec_env.reset()

    # pause on space
    pause_key = ord(' ')
    if pause_key in keys and keys[pause_key] & p.KEY_WAS_TRIGGERED:
        paused = not paused
        print('paused' if paused else 'unpaused')

    # quit on q
    q_key = ord('q')
    if q_key in keys and keys[q_key] & p.KEY_WAS_TRIGGERED:
        print('quitting...')
        break

vec_env.close()
