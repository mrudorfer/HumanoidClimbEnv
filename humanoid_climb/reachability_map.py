from typing import Union

import numpy as np
import pybullet
from scipy.spatial.transform import Rotation

from humanoid_climb.assets.humanoid import Humanoid


def visualize_slice(bullet_client, coords, vals):
    if coords is not None:
        max_val = np.max(vals)
        min_val = np.min(vals)

        for i in range(len(coords)):
            c = (vals[i] - min_val) / (max_val - min_val)
            visual_shape = bullet_client.createVisualShape(pybullet.GEOM_SPHERE, radius=0.05, rgbaColor=[1 - c, c, 0, 0.5])
            bullet_client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, basePosition=coords[i])


class Single3DMap:
    def __init__(self, lower_bounds, upper_bounds, voxel_res, no_map=False):
        # get dimensions
        self.n_bins_x = int(np.ceil((upper_bounds[0] - lower_bounds[0]) / voxel_res))
        self.n_bins_y = int(np.ceil((upper_bounds[1] - lower_bounds[1]) / voxel_res))
        self.n_bins_z = int(np.ceil((upper_bounds[2] - lower_bounds[2]) / voxel_res))

        self.lower_bounds = np.asarray(lower_bounds)
        self.upper_bounds = np.asarray(upper_bounds)
        self.voxel_res = voxel_res

        if no_map:
            self.map = None
        else:
            self.map = np.zeros(
                shape=(self.n_bins_x, self.n_bins_y, self.n_bins_z), dtype=np.float32
            )

        # create coordinate representation of voxel centers and cache it for fast access
        x, y, z = np.meshgrid(
            np.arange(self.n_bins_x) * self.voxel_res + self.lower_bounds[0] + self.voxel_res/2,
            np.arange(self.n_bins_y) * self.voxel_res + self.lower_bounds[1] + self.voxel_res/2,
            np.arange(self.n_bins_z) * self.voxel_res + self.lower_bounds[2] + self.voxel_res/2,
            indexing='ij'
        )
        coords = np.vstack([x.ravel(), y.ravel(), z.ravel()])  # shape (3, n)
        self._coords_homogenuous = np.vstack([coords, np.ones((1, coords.shape[1]))])

    def to_dict(self) -> dict:
        d = {
            'map': self.map,
            'lower_bounds': self.lower_bounds,
            'upper_bounds': self.upper_bounds,
            'voxel_res': self.voxel_res
        }
        return d

    @classmethod
    def from_dict(cls, d: dict):
        rm = cls(
            lower_bounds=d['lower_bounds'],
            upper_bounds=d['upper_bounds'],
            voxel_res=d['voxel_res'],
            no_map=True
        )
        rm.map = d['map']
        return rm

    def _get_dim_index(self, dim: Union[int, str], value: float) -> int:
        """
        For a given value in a certain dimension, returns the corresponding index in the map.
        :param dim: [int | str] 0/'x', 1/'y', 2/'z'
        :param value: float, the value

        :return: int, index
        """
        if dim in [0, 'x']:
            lower_limit = self.lower_bounds[0]
            n_bins = self.n_bins_x
        elif dim in [1, 'y']:
            lower_limit = self.lower_bounds[1]
            n_bins = self.n_bins_y
        elif dim in [2, 'z']:
            lower_limit = self.lower_bounds[2]
            n_bins = self.n_bins_z
        else:
            raise ValueError(f'invalid dim: {dim}')

        idx = int((value - lower_limit) / self.voxel_res)
        if idx < 0:
            raise IndexError(f'idx < 0 -- val: {value}, dim: {dim}')
        if idx >= n_bins:
            raise IndexError(f'xy idx too large -- val: {value}, dim: {dim}')
        return idx

    def get_indices_for_ee_pos(self, ee_pos):
        """
        Gets the map indices as tuple for a given end-effector position.
        :param ee_pos: [x, y, z] position of end effector

        :returns: tuple, indices of map
        """
        x_idx = self._get_dim_index('x', ee_pos[0])
        y_idx = self._get_dim_index('y', ee_pos[1])
        z_idx = self._get_dim_index('z', ee_pos[2])

        return x_idx, y_idx, z_idx

    @property
    def shape(self):
        return self.map.shape

    def add_reachability_value(self, indices, value, method='max'):
        assert method in ['max', 'sum', 'min', 'replace']

        if method == 'replace':
            self.map[indices] = value
        elif method == 'sum':
            self.map[indices] += value
        elif method == 'min':
            self.map[indices] = np.minimum(self.map[indices], value)
        elif method == 'max':
            self.map[indices] = np.maximum(self.map[indices], value)

    def get_reachability_value(self, indices):
        return self.map[indices]

    def get_coordinate_map(self, pos=None, orn=None, skip_zero=True):
        """
        retrieve map as coordinates, possibly transforming it to new position/orientation of robot base
        :param pos: [x, y, z] robot base position
        :param orn: [x, y, z, w] robot base orientation
        :param skip_zero: bool, True skips coordinates with value of zero
        :return: Tuple((n, 3), (n)), coordinates and values
        """
        if pos is None:
            pos = [0, 0, 0]
        if orn is None:
            orn = [0, 0, 0, 1]

        tf = np.eye(4)
        tf[:3, :3] = Rotation.from_quat(orn).as_matrix()
        tf[:3, 3] = pos

        transformed_coords = tf @ self._coords_homogenuous
        transformed_coords = transformed_coords[:3, :].T  # (n, 3)
        values = self.map.ravel()

        if skip_zero:
            mask = values > 0
            return transformed_coords[mask], values[mask]

        return transformed_coords, values


    def get_slice(self, pos=None, orn=None, x=0.4, epsilon=None):
        """
        The voxel grid will be transformed according to pos and orn, then it will be sliced by a plane at x.
        We take all voxel centres that are closer than epsilon to x, and project them onto the plane.

        :param pos: [x, y, z] position of climber base
        :param orn: [x, y, z, w] quaternion orientation of climber base
        :param x: float, where to slice, defaults to 0.4 (standard wall position)
        :param epsilon: float, voxel centers with x - epsilon < v_x < x + epsilon are considered to be part of the slice
                        defaults to half of voxel_res to ensure at least one voxel is in

        :return: Tuple((n, 3), (n,)) array of coordinates with corresponding reachability value, or (None, None)
        """
        if pos is None:
            pos = [0, 0, 0]
        if orn is None:
            orn = [0, 0, 0, 1]
        if epsilon is None:
            epsilon = self.voxel_res / 2

        coords, values = self.get_coordinate_map(pos, orn)
        mask = np.logical_and(coords[:, 0] > x-epsilon, coords[:, 0] < x+epsilon)

        if not np.any(mask):
            return None, None

        coords, values = coords[mask], values[mask]
        coords[:, 0] = x
        return coords, values



class ReachabilityMap:
    def __init__(self, voxel_res=0.1, no_maps=False):
        if no_maps:
            self.left_hand = None
            self.right_hand = None
            self.left_foot = None
            self.right_foot = None

        else:
            # initialise maps based on reachability of climber
            # extreme values from 100k samples:
            # left_hand:	min (x, y, z): [-0.4735273  -0.16417934 -0.46705676]	max (x, y, z): [0.6582912  0.81802635 0.67625497]
            # right_hand:	min (x, y, z): [-0.47955246 -0.81793345 -0.46738464]	max (x, y, z): [0.6578822  0.16117683 0.67533535]
            # left_foot:	min (x, y, z): [-0.86239196 -0.85086985 -1.12218449]	max (x, y, z): [1.02577562 0.9835361  0.61376257]
            # right_foot:	min (x, y, z): [-0.88795972 -0.99322471 -1.12176351]	max (x, y, z): [1.02297985 0.90054887 0.61948916]
            self.left_hand = Single3DMap(
                lower_bounds=[-0.55, -0.25, -0.55],
                upper_bounds=[ 0.70,  0.90,  0.75],
                voxel_res=voxel_res
            )
            self.right_hand = Single3DMap(
                lower_bounds=[-0.55, -0.90, -0.55],
                upper_bounds=[ 0.70,  0.25,  0.75],
                voxel_res=voxel_res
            )
            self.left_foot = Single3DMap(
                lower_bounds=[-0.95, -0.95, -1.20],
                upper_bounds=[ 1.10,  1.05,  0.70],
                voxel_res=voxel_res
            )
            self.right_foot = Single3DMap(
                lower_bounds=[-0.95, -1.05, -1.20],
                upper_bounds=[ 1.10,  0.95,  0.70],
                voxel_res=voxel_res
            )

        self.maps = None
        self.init_map_dict()

    def init_map_dict(self):
        self.maps = {
            'left_hand': self.left_hand,
            'right_hand': self.right_hand,
            'left_foot': self.left_foot,
            'right_foot': self.right_foot
        }


    def to_file(self, filename):
        save_dict = {
            'left_hand': self.left_hand.to_dict(),
            'right_hand': self.right_hand.to_dict(),
            'left_foot': self.left_foot.to_dict(),
            'right_foot': self.right_foot.to_dict(),
        }
        np.savez(filename, **save_dict)

    def print_info(self):
        print('*********************')
        print(self.__class__.__name__)
        mem = 0.0
        for key, rmap in self.maps.items():
            print(key)
            print(f'\tlower bounds: {rmap.lower_bounds}')
            print(f'\tupper bounds: {rmap.upper_bounds}')
            print(f'\tn bins: ({rmap.n_bins_x}, {rmap.n_bins_y}, {rmap.n_bins_z}), voxel res: {rmap.voxel_res}')
            print(f'\t{np.count_nonzero(rmap.map)}/{rmap.map.size} cells occupied ({np.count_nonzero(rmap.map)/rmap.map.size:.2f})')
            print(f'\tmemory required: {rmap.map.nbytes / 1024:.2f}kB')
            mem += rmap.map.nbytes
        print(f'total memory required: {mem / 1024:.2f}kB')
        print('*********************')

    @classmethod
    def from_file(cls, filename):
        rm = cls(no_maps=True)
        data = dict(np.load(filename, allow_pickle=True))
        rm.left_hand = Single3DMap.from_dict(data['left_hand'].item())
        rm.right_hand = Single3DMap.from_dict(data['right_hand'].item())
        rm.left_foot = Single3DMap.from_dict(data['left_foot'].item())
        rm.right_foot = Single3DMap.from_dict(data['right_foot'].item())
        rm.init_map_dict()
        print(f'{cls.__name__} loaded from {filename}')
        rm.print_info()
        return rm

    def record_manipulability_values(self, climber: Humanoid):
        m = climber.get_manipulability(use_rotation=True)
        for i, (key, rmap) in enumerate(self.maps.items()):
            pos = climber.effectors[i].current_position()
            idcs = rmap.get_indices_for_ee_pos(pos)
            rmap.add_reachability_value(idcs, m[i], method='max')
            # rmap.add_reachability_value(idcs, 1.0, method='sum')
