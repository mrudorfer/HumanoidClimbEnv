import time

import pybullet as p
from pybullet_utils.bullet_client import BulletClient
import numpy as np
from tqdm import tqdm

from humanoid_climb.assets.humanoid import Humanoid
from humanoid_climb.climbing_config import ClimbingConfig
from humanoid_climb.reachability_map import ReachabilityMap


def create_rmap(n=1_000_000, fn='./maps/rmap.npz'):
    bullet_client = BulletClient(p.DIRECT)
    config = ClimbingConfig('./configs/sim_config.json')
    config.climber['position'] = [0, 0, 0]
    climber = Humanoid(bullet_client, config.climber, fixedBase=True)

    rmap = ReachabilityMap()
    rng = np.random.default_rng()
    for i in tqdm(range(n)):
        climber.set_random_joint_config(rng)
        rmap.record_manipulability_values(climber)

    coords, vals = rmap.left_foot.get_coordinate_map()
    max_val = np.max(vals)
    min_val = np.min(vals)
    print('max manipulability:', max_val)
    print('mean value:', np.mean(vals))
    print('min value:', min_val)
    print('n vals:', len(vals))
    print('map size:', rmap.left_foot.map.shape, rmap.left_foot.map.size)

    rmap.to_file(fn)
    del rmap
    rmap = ReachabilityMap.from_file(fn)
    coords, vals = rmap.left_foot.get_coordinate_map()
    max_val = np.max(vals)
    min_val = np.min(vals)
    print('max manipulability:', max_val)
    print('mean value:', np.mean(vals))
    print('min value:', min_val)
    print('n vals:', len(vals))
    print('map size:', rmap.left_foot.map.shape, rmap.left_foot.map.size)

    bullet_client.disconnect()



def visualise_slice():
    bullet_client = BulletClient(p.GUI)
    config = ClimbingConfig('./configs/sim_config.json')
    config.climber['position'] = [0, 0, 0]
    climber = Humanoid(bullet_client, config.climber, fixedBase=True)

    rmap = ReachabilityMap.from_file('maps/rmap.npz')

    for limb, limb_map in rmap.maps.items():
        print('limb:', limb)

        coords, vals = limb_map.get_slice()
        if coords is not None:
            max_val = np.max(vals)
            min_val = np.min(vals)

            for i in range(len(coords)):
                c = (vals[i]-min_val) / (max_val-min_val)
                visual_shape = bullet_client.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1-c, c, 0, 0.5])
                bullet_client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, basePosition=coords[i])

        input()
    bullet_client.disconnect()


def find_max_values(n=1000):
    bullet_client = BulletClient(p.DIRECT)
    config = ClimbingConfig('./configs/sim_config.json')
    config.climber['position'] = [0, 0, 0]
    climber = Humanoid(bullet_client, config.climber, fixedBase=True)

    values = {}
    effector_keys = config.climber['end_effectors']
    for key in effector_keys:
        values[key] = np.zeros(shape=(n, 3))

    rng = np.random.default_rng()
    for i in tqdm(range(n)):
        climber.set_random_joint_config(rng)
        for j in range(len(effector_keys)):
            values[effector_keys[j]][i] = climber.effectors[j].current_position()

    for key, positions in values.items():
        max = np.max(positions, axis=0)
        min = np.min(positions, axis=0)
        print(f'{key}:\tmin (x, y, z): {min}\tmax (x, y, z): {max}')

    bullet_client.disconnect()


if __name__ == '__main__':
    # ReachabilityMap.from_file('maps/rmap.npz')
    # visualise_slice()
    # check_save_load()
    create_rmap(n=10_000_000, fn='./maps/rmap_10M.npz')
