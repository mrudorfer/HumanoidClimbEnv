import os
import warnings
from typing import Optional, Union, List, Dict, Any
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization


def linear_schedule(initial_value, final_value=0.0):
    def func(progress):
        # progress decreases from 1.0 to 0.0
        new_value = progress * (initial_value - final_value) + final_value
        return new_value
    return func


def step_schedule(initial_value):
    def func(progress):
        if progress > 0.5:
            return initial_value
        if progress > 0.25:
            return initial_value / 2
        else:
            return initial_value / 4
    return func

class UpdateInitStatesCallback(BaseCallback):
    def __init__(self, env: Union[gym.Env, VecEnv], state_fn: str, verbose: int = 0):
        super().__init__(verbose)
        self.added_states_count = 0

        # convert env to VecEnv for consistency
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        self.vec_env = env
        self.state_fn = state_fn

        self.rng = np.random.default_rng()
        self.start_prob = 1.0  # 0.5
        self.min_prob = 0.01
        self.fade_out = 50_000  # 25_000

    def _on_step(self) -> bool:
        # nothing to be done here
        return True

    def _on_rollout_end(self) -> None:
        data = self.vec_env.env_method('pop_collected_states')  # todo: gives warning, but no idea how to do properly
        collected_states = []
        collected_stances = []

        for elem in data:
            states, stances = elem['states'], elem['stances']
            if states and stances:
                collected_states.extend(states)
                collected_stances.extend(stances)

        if collected_states and collected_stances:
            collected_states = np.stack(collected_states)
            collected_stances = np.stack(collected_stances)

            # calculate probability of adding new states
            progress = max(0.0, 1.0 * (self.fade_out - self.added_states_count) / self.fade_out)  # 1 initially, goes down to 0
            prob = progress * (self.start_prob - self.min_prob) + self.min_prob
            select_idx = self.rng.uniform(size=len(collected_states)) < prob

            if self.verbose >= 1:
                print(f'Collecting states...')
                print(f'All collected states {collected_states.shape} and stances {collected_stances.shape}')
                print(f'Already added states {self.added_states_count}/{self.fade_out}; progress {progress:.4f}')
                print(f'Probability is {prob:.4f}, selection ratio {np.sum(select_idx)/len(collected_states):.4f}. selected {np.sum(select_idx)} out of {len(collected_states)}')

            if np.any(select_idx):
                data = dict(np.load(self.state_fn, allow_pickle=True))
                init_states = data['states']
                init_stances = data['stances']
                data['states'] = np.concatenate([init_states, collected_states[select_idx]])
                data['stances'] = np.concatenate([init_stances, collected_stances[select_idx]])
                np.savez(self.state_fn, **data)

                self.added_states_count += np.sum(select_idx)
                self.vec_env.env_method('reload_init_states')

        self.logger.record("curriculum/added_states_count", self.added_states_count)


class CustomEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.
    Adds to the original SB3 EvalCallback the saving of VecNormalize stats.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])  # type: ignore[list-item, return-value]

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results: List[List[float]] = []
        self.evaluations_timesteps: List[int] = []
        self.evaluations_length: List[List[int]] = []
        # For computing success rate
        self._is_success_buffer: List[bool] = []
        self.evaluations_successes: List[List[bool]] = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    # MR: added saving the VecNormalize features
                    if self.model.get_vec_normalize_env() is not None:
                        path = os.path.join(self.best_model_save_path, 'vecnormalize.pkl')
                        self.model.get_vec_normalize_env().save(path)
                        if self.verbose > 1:
                            print(f"Saving VecNormalize to {path}")
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)
