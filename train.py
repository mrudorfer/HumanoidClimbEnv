import time
from datetime import timedelta

import gymnasium as gym
import pybullet as p
import stable_baselines3 as sb
import os
import shutil
import argparse

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from torch.backends.mkl import verbose

import wandb
from wandb.integration.sb3 import WandbCallback
import humanoid_climb.stances as stances
from humanoid_climb.climbing_config import ClimbingConfig
from callbacks import CustomEvalCallback, UpdateInitStatesCallback, linear_schedule

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def make_env(env_id: str, rank: int, config, seed: int = 0, max_steps: int = 1000) -> gym.Env:
	def _init():
		env = gym.make(env_id, config=config, render_mode=None, max_ep_steps=max_steps)
		m_env = Monitor(env)
		m_env.reset(seed=seed + rank)
		return m_env

	set_random_seed(seed)
	return _init


def train(env_name, sb3_algo, workers, n_steps, episode_steps, path_to_model=None):
	config = {
		"policy_type": "MlpPolicy",
		"total_timesteps": n_steps,
		"env_name": env_name,
	}
	run = wandb.init(
		project="HumanoidClimbMulti",
		config=config,
		sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
		monitor_gym=False,  # auto-upload the videos of agents playing the game
		save_code=False,  # optional
	)

	climbing_config = ClimbingConfig(
		'./configs/sim_config.json', './configs/simple_train_config.json')
	max_ep_steps = episode_steps
	gamma = 0.995
	save_path = f"{model_dir}/{run.id}"

	# copy init states file and put it into the model dir
	os.makedirs(save_path, exist_ok=True)
	new_init_state_fn = os.path.join(save_path, 'init_states.npz')
	shutil.copy(climbing_config.init_states_fn, new_init_state_fn)
	climbing_config.init_states_fn = new_init_state_fn

	# create envs
	vec_env = SubprocVecEnv(
		[make_env(env_name, i, climbing_config, max_steps=max_ep_steps) for i in range(workers)],
		start_method="spawn"
	)
	vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=gamma)


	eval_callback = CustomEvalCallback(vec_env, best_model_save_path=f"{save_path}/models/", log_path=f"{save_path}/logs/",
									   eval_freq=500, deterministic=True, render=False)

	cust_callback = UpdateInitStatesCallback(vec_env, climbing_config.init_states_fn, verbose=2)

	if sb3_algo == 'PPO':
		if path_to_model is None:
			policy_kwargs = dict(net_arch=[256, 256, 256])
			model = sb.PPO('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir,
						   batch_size=2048, n_epochs=5, ent_coef=0.001, gamma=gamma, clip_range=0.2,
						   n_steps=2048, learning_rate=linear_schedule(0.0005, 0.000001),
						   policy_kwargs=policy_kwargs)
		else:
			model = sb.PPO.load(path_to_model, env=vec_env)
	elif sb3_algo == 'SAC':
		if path_to_model is None:
			model = sb.SAC('MlpPolicy', vec_env, verbose=1, device='cuda', tensorboard_log=log_dir)
		else:
			model = sb.SAC.load(path_to_model, env=vec_env)
	else:
		print('Algorithm not found')
		return

	model.learn(
		total_timesteps=config["total_timesteps"],
		progress_bar=True,
		callback=[WandbCallback(
			gradient_save_freq=5000,
			model_save_freq=5000,
			model_save_path=save_path,
			verbose=2,
		), eval_callback, cust_callback],
	)
	run.finish()


def test(env, sb3_algo, path_to_model):
	if sb3_algo == 'SAC':
		model = sb.SAC.load(path_to_model, env=env)
	elif sb3_algo == 'TD3':
		model = sb.TD3.load(path_to_model, env=env)
	elif sb3_algo == 'A2C':
		model = sb.A2C.load(path_to_model, env=env)
	elif sb3_algo == 'DQN':
		model = sb.DQN.load(path_to_model, env=env)
	elif sb3_algo == 'PPO':
		model = sb.PPO.load(path_to_model, env=env)
	else:
		print('Algorithm not found')
		return

	vec_env = model.get_env()
	obs = vec_env.reset()
	score = 0
	step = 0

	while True:
		action, _state = model.predict(obs, deterministic=True)
		obs, reward, done, info = vec_env.step(action)
		score += reward
		step += 1

		# env.reset() auto called on vec_env?
		if done:
			print(f"Episode Over, Score: {score}, Steps {step}")
			score = 0
			step = 0

		# Reset on backspace
		keys = p.getKeyboardEvents()
		if 114 in keys and keys[114] & p.KEY_WAS_TRIGGERED:
			score = 0
			step = 0
			env.reset()

	env.close()


if __name__ == '__main__':

	# Parse command line inputs
	parser = argparse.ArgumentParser(description='Train or test model.')
	parser.add_argument('--gymenv', type=str, default='HumanoidClimb-v0', help='Gymnasium environment i.e. Humanoid-v4')
	parser.add_argument('--sb3_algo', type=str, choices=['PPO', 'SAC'], default='PPO', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
	parser.add_argument('-w', '--workers', type=int, default=int(24))
	parser.add_argument('-t', '--train', action='store_true')
	parser.add_argument('-f', '--file', required=False, default=None)
	parser.add_argument('-s', '--test', metavar='path_to_model')
	parser.add_argument('-n', '--n_steps', type=int, default=int(100_000_000))
	parser.add_argument('-e', '--episode_steps', type=int, default=int(200))

	args = parser.parse_args()

	start_time = time.time()

	if args.train:
		if args.file is None:
			print(f'<< Training from scratch! >>')
			train(args.gymenv, args.sb3_algo, args.workers, args.n_steps, args.episode_steps)
		elif os.path.isfile(args.file):
			print(f'<< Continuing {args.file} >>')
			train(args.gymenv, args.sb3_algo, args.workers, args.n_steps, args.episode_steps, args.file)

	if args.test:
		if os.path.isfile(args.test):
			stances.set_root_path("./humanoid_climb")
			stance = stances.STANCE_14_1
			max_steps = 600

			env = gym.make(args.gymenv, render_mode='human', max_ep_steps=max_steps, **stance.get_args())
			test(env, args.sb3_algo, path_to_model=args.test)
		else:
			print(f'{args.test} not found.')

	duration = timedelta(seconds=time.time() - start_time)
	print(f'Completed the process. Duration: {duration}')
