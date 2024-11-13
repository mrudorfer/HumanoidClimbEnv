import os.path

import gymnasium as gym
import pybullet as p
from stable_baselines3 import PPO
from humanoid_climb.climbing_config import ClimbingConfig


# load config, create and reset env, load policy
config = ClimbingConfig('./configs/config.json')
env = gym.make('HumanoidClimb-v0',
               render_mode='human',
               max_ep_steps=10000000,
               config=config)
obs, info = env.reset()
policies = [
    '1_10_9_n_n.zip',
    '2_10_9_2_n.zip',
    '3_10_9_2_1.zip',
    '4_10_13_2_1.zip',
    '5_10_13_2_5.zip',
    '6_13_13_n_5.zip',
    '7_13_13_6_5.zip',
    '8_14_13_6_5.zip',
    '9_14_17_6_5.zip',
    '10_14_17_n_9.zip',
    '11_14_17_10_9.zip',
    '12_18_17_10_9.zip',
    '13_18_20_10_9.zip',
    '14_20_20_10_9.zip',
]
policy_dir = './humanoid_climb/models/'
model = PPO.load(os.path.join(policy_dir, policies[0]), env=env)
policy_idx = 0

# prepare variables for the while loop
terminated, truncated = False, False
score, step = 0, 0
paused = False

print('In PyBullet window, press:')
print('\tr          reset episode')
print('\tspacebar   pause episode')
print('\tq          quit')

while True:
    if not paused:
        # use policy to predict next action, then step environment
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        step += 1

    if (step % 200 == 0) and not paused:
        print(f'step {step}: score {score}')

    # if episode terminates, pause (user needs to reset)
    # note that episode currently does not terminate after goal has successfully been reached
    if (terminated or truncated) and not paused:
        print(f'terminated after {step} steps. score: {score}')
        paused = True

    if step > 1 and not paused and info['stance_reached']:
        policy_idx += 1
        print(f'stance reached after {step} steps!! activating policy {policies[policy_idx][:-4]}.')
        model = PPO.load(os.path.join(policy_dir, policies[policy_idx]), env=env)

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
        policy_idx = 0
        model = PPO.load(os.path.join(policy_dir, policies[policy_idx]), env=env)
        obs, info = env.reset()

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

env.close()
