## installation

```cmd
conda create -n climb python=3.10
conda activate climb
conda install numpy pybullet gymnasium stable-baselines3=2.3 wandb tensorboard tqdm rich -c conda-forge
```


## todo

- define how to use environment:
  - one to many pre-determined stance transitions
  - option to update the future stance transitions dynamically
  - initialise either in T-pose, or in a given stance and state
- clean environment:
  - step after applying actions
  - multiple steps?
- use a climber without grasping actions
  - grasping actions are predefined
  - consider a transition period for resting at a stance
  - i.e., episode only ends when stance has been reached for 20 timesteps
  - provide further reward by using torque-based loss
  - perhaps also 20 first timesteps of each episode still has limbs attached, to allow for acceleration
