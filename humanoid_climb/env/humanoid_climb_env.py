from collections.abc import Iterable, Sized

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

from typing import Optional
from pybullet_utils.bullet_client import BulletClient
from humanoid_climb.assets.humanoid import Humanoid
from humanoid_climb.assets.asset import Asset


class HumanoidClimbEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, config, render_mode: Optional[str] = None, max_ep_steps: Optional[int] = 602,
                 state_file: Optional[str] = None):
        # general configuration
        self.np_random, _ = gym.utils.seeding.np_random()
        self.config = config
        self.render_mode = render_mode
        self.max_ep_steps = max_ep_steps


        # 17 joint actions + no grasp actions
        self.action_space = gym.spaces.Box(-1, 1, (17,), np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._get_obs_dim(),), dtype=np.float32)

        # configure pybullet GUI and load environment
        if self.render_mode == 'human':
            self._p = BulletClient(p.GUI)
            self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            self._p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=-90, cameraPitch=0, cameraTargetPosition=[0, 0, 3])
        else:
            self._p = BulletClient(p.DIRECT)

        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.8)
        self.floor = self._p.loadURDF("plane.urdf")
        self.floor = Asset(self._p, self.config.plane)
        self.wall = Asset(self._p, self.config.surface)
        self.climber = Humanoid(self._p, self.config.climber)

        self.targets = dict()
        for key in self.config.holds:
            self.targets[key] = Asset(self._p, self.config.holds[key])
            self._p.addUserDebugText(text=key, textPosition=self.targets[key].body.initialPosition, textSize=0.7, lifeTime=0.0, textColorRGB=[0.0, 0.0, 1.0])
        self.climber.targets = self.targets

        self.debug_stance_text = self._p.addUserDebugText(text=f"", textPosition=[0, 0, 0], textSize=1, lifeTime=0.1, textColorRGB=[1.0, 0.0, 1.0])

        # initialization of variables
        self.motion_path = [self.config.stance_path[stance]['desired_holds'] for stance in self.config.stance_path]
        self.init_states = None
        if state_file is not None:
            self.init_states = np.load(state_file, allow_pickle=True)['arr_0']
        self.sim_steps_per_action = config.sim_steps_per_action

        self.steps = 0
        self.current_stance = []
        self.desired_stance = []
        self.limbs_transitioning = []
        self.desired_stance_index = 0
        self.grasp_status = []  # between 0 and 1, 1 meaning fully grasp and 0 is released.
        self.grasp_actions = []  # -1 (release), 0 (stay as is), 1 (grasp)


    def step(self, action):
        # apply torque actions
        self.climber.apply_torque_actions(action)

        # step simulation
        for _ in range(self.sim_steps_per_action):
            self._p.stepSimulation()
        self.steps += 1

        # connect climber to holds and update stance
        self.grasp_holds()
        self.update_grasp_status(grasp_step=1.0, release_step=0.05)
        self.update_stance()

        # determine if next stance and overall goal are reached
        grasp_actions_done = np.sum(self.grasp_actions) == 0
        stance_reached = (self.current_stance == self.desired_stance) and grasp_actions_done
        goal_reached = (self.current_stance == self.motion_path[-1]) and grasp_actions_done

        # advance to next stance transition
        if stance_reached and not goal_reached:
            self.set_next_desired_stance()
            # set action to -1 if plan to release that hold to transition
            self.grasp_actions = -1 * np.array(np.bitwise_and(self.limbs_transitioning, self.current_stance != -1))

        ob = self._get_obs()
        info = {
            'is_success': goal_reached,
            'stance_reached': stance_reached
        }

        # reward = self.calculate_reward_eq1()
        reward = self.calculate_reward_negative_distance()
        if goal_reached:
            reward += 1000

        terminate = goal_reached or (self.is_on_floor())
        truncate = self.steps >= self.max_ep_steps

        return ob, reward, terminate, truncate, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.current_stance = self.motion_path[0]

        self.climber.reset()
        if self.init_states is not None:
            idx = self.np_random.choice(len(self.init_states))
            self.climber.set_state(self.init_states[idx], self.current_stance)

        self.desired_stance_index = 0
        self.set_next_desired_stance()
        # set grasp status to 1 if a hold is grasped; set action to -1 if plan to release that hold to transition
        self.grasp_status = np.array([1.0 if c != -1 else 0.0 for c in self.current_stance])
        self.grasp_actions = -1 * np.array(np.bitwise_and(self.limbs_transitioning, self.current_stance != -1))
        # release actions are only valid if we are actually on a hold currently
        for i in range(len(self.current_stance)):
            if self.current_stance == -1:
                self.grasp_actions[i] = 0

        ob = self._get_obs()
        info = {
            'is_success': False,
            'stance_reached': False
        }

        if self.render_mode == 'human':
            for key in self.targets:
                colour = [0.0, 0.7, 0.1, 0.75] if key in self.desired_stance else [1.0, 0, 0, 0.75]
                self._p.changeVisualShape(objectUniqueId=self.targets[key].id, linkIndex=-1, rgbaColor=colour)

        return np.array(ob, dtype=np.float32), info

    def calculate_reward_negative_distance(self):
        distances = [0.0]
        eff_positions = [eff.current_position() for eff in self.climber.effectors]
        for i, c_stance in enumerate(self.desired_stance):
            if c_stance == -1:
                # no target! current position is fine
                pass
            else:
                # get distance to target
                eff_target = self.targets[c_stance].body.initialPosition
                distances.append(np.linalg.norm(eff_target - np.array(eff_positions[i])))

        reward = np.clip(-1.0 * np.sum(distances), -2.0, float('inf'))
        # reward += 1000 if self.current_stance == self.desired_stance else 0

        if self.is_on_floor():
            # episode will be terminated, i.e., put maximum punishment for all remaining steps
            reward += (self.max_ep_steps - self.steps) * -2.0

        return reward

    def set_next_desired_stance(self):
        # update stance and target info
        self.desired_stance_index += 1
        self.desired_stance = self.motion_path[self.desired_stance_index]

        # check which limbs are transitioning
        self.limbs_transitioning = [hold1 != hold2 for hold1, hold2 in zip(self.desired_stance, self.current_stance)]

        # update visualisation
        if self.render_mode == 'human':
            # reset previous desired target colours to red
            for key in self.current_stance:
                if key == -1: continue
                self._p.changeVisualShape(objectUniqueId=self.targets[key].id, linkIndex=-1,
                                          rgbaColor=[1.0, 0.0, 0.0, 0.75])

            # set new desired targets to green
            for key in self.desired_stance:
                if key == -1: continue
                self._p.changeVisualShape(objectUniqueId=self.targets[key].id, linkIndex=-1,
                                          rgbaColor=[0.0, 0.7, 0.1, 0.75])

    def grasp_holds(self):
        for i, (transitioning, target) in enumerate(zip(self.limbs_transitioning, self.desired_stance)):
            if transitioning and target != -1:
                attached = self.climber.attempt_attach_eff_to_hold(i, target)
                if attached:
                    self.limbs_transitioning[i] = False
                    self.grasp_actions[i] = 1

    def update_grasp_status(self, release_step = 0.2, grasp_step = 1.0):
        # steps need to be 0 < step <= 1.0
        # the smaller the value, the more gradually holds are being grasped/released (takes more timesteps)
        # if the value is 1.0, holds are grasped/released immediately.

        # apply grasp action
        for i, action in enumerate(self.grasp_actions):
            if action == 1:
                # hold is already attached, just need to complete the action (i.e. waiting some timesteps)
                self.grasp_status[i] = self.grasp_status[i] + grasp_step
                if self.grasp_status[i] >= 1.0:
                    self.grasp_status[i] = 1.0
                    self.grasp_actions[i] = 0
            if action == -1:
                # need to wait until action completed to actually detach from the hold
                self.grasp_status[i] = self.grasp_status[i] - release_step
                if self.grasp_status[i] <= 0.0:
                    self.grasp_status[i] = 0.0
                    self.grasp_actions[i] = 0
                    self.climber.detach(i)

    def update_stance(self):
        self.current_stance = self.climber.effector_attached_to
        self.limbs_transitioning = [hold1 != hold2 for hold1, hold2 in zip(self.desired_stance, self.current_stance)]

        if self.render_mode == 'human':
            torso_pos = self.climber.robot_body.current_position()
            torso_pos[1] += 0.15
            torso_pos[2] += 0.35
            self.debug_stance_text = self._p.addUserDebugText(text=f"{self.current_stance}", textPosition=torso_pos,
                                                              textSize=1, lifeTime=0.1, textColorRGB=[1.0, 0.0, 1.0],
                                                              replaceItemUniqueId=self.debug_stance_text)

    def get_stance_center(self, stance):
        """
        Determines the xyz center of a stance as average of all holds. Free limbs are ignored.
        :param stance: list of hold keys
        :returns: ndarray(3,), xyz center position
        """
        hold_positions = []
        for key in stance:
            if key != -1:
                pos = self.targets[key].body.initialPosition
                # pos, _ = self._p.getBasePositionAndOrientation(self.targets[key].id)
                hold_positions.append(pos)
        center = np.asarray(hold_positions).mean(axis=0)
        return center

    def visualise_climber_pos(self):
        pos, orn = self._p.getBasePositionAndOrientation(self.climber.robot)
        visual_shape = self._p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[1, 0, 0, 1])
        body_id = self._p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, basePosition=pos)

    def _get_obs_dim(self):
        return 7 + 17 * 2 + 17 * 13 + 4 * 9  # = 271

    def _get_obs(self):
        # variant that translates all values wrt to a stance center
        obs = np.empty(self._get_obs_dim(), dtype=np.float32)
        idx = 0

        def add_to_obs(data):
            nonlocal idx
            _n = 1
            if isinstance(data, Sized):
                _n = len(data)
            obs[idx:idx+_n] = data
            idx += _n

        # we compute a stance center and all positions will be expressed relative to that
        base_pos = self.get_stance_center(self.desired_stance)
        trunk_pos, trunk_orn = self._p.getBasePositionAndOrientation(self.climber.robot)
        trunk_pos = np.asarray(trunk_pos) - base_pos
        add_to_obs(trunk_pos)
        add_to_obs(trunk_orn)

        # add joint angles
        for joint_name, joint in self.climber.joints.items():
            add_to_obs(joint.get_state())  # pos and vel

        # for each link (17):
        # position and orientation relative to torso (base) = 3 + 4
        # linear and angular velocity                       = 3 + 3 (euler?)
        states = self._p.getLinkStates(self.climber.robot,
                                       linkIndices=[joint.jointIndex for joint in self.climber.ordered_joints],
                                       computeLinkVelocity=1)
        for s, state in enumerate(states):
            world_pos, world_orn, _, _, _, _, linear_vel, ang_vel = state
            relative_pos = world_pos - base_pos
            # relative_orn = self._p.getDifferenceQuaternion(world_orn, base_orn)

            add_to_obs(relative_pos)
            add_to_obs(world_orn)
            add_to_obs(linear_vel)
            add_to_obs(ang_vel)

        # for each limb (effector): - 9 x 4 = 36
        # current xzy position of limb should already be contained in the above
        # xyz position of target hold; vector to hold; distance to hold; grasp action and grasp status
        eff_positions = [eff.current_position()-base_pos for eff in self.climber.effectors]
        for i, c_stance in enumerate(self.desired_stance):
            if c_stance == -1:
                # no target! just assume current position is fine instead of giving arbitrary values
                eff_target = eff_positions[i]
            else:
                eff_target = self.targets[c_stance].body.initialPosition - base_pos

            translation = eff_target - np.array(eff_positions[i])
            dist = np.linalg.norm(translation)

            add_to_obs(eff_target)
            add_to_obs(translation)
            add_to_obs(dist)
            add_to_obs(self.grasp_actions[i])
            add_to_obs(self.grasp_status[i])

        assert idx == self._get_obs_dim(), f'did not fill observation space. idx: {idx}. dim: {self._get_obs_dim()}'
        return obs

    def is_on_floor(self):
        floor_contact = self._p.getContactPoints(bodyA=self.climber.robot, bodyB=self.floor.id)
        for i in range(len(floor_contact)):
            contact_body = floor_contact[i][3]
            exclude_list = [self.climber.parts["left_foot"].bodyPartIndex, self.climber.parts["right_foot"].bodyPartIndex]
            if contact_body not in exclude_list:
                return True

        return False

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def visualise_reward(self, reward, min, max):
        if self.render_mode != 'human': return
        value = np.clip(reward, min, max)
        normalized_value = (value - min) / (max - min) * (1 - 0) + 0
        colour = [0.0, normalized_value / 1.0, 0.0, 1.0] if reward > 0.0 else [normalized_value / 1.0, 0.0, 0.0, 1.0]
        self._p.changeVisualShape(objectUniqueId=self.climber.robot, linkIndex=-1, rgbaColor=colour)

    def close(self):
        self._p.disconnect()
