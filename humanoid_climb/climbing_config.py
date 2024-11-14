import json
from pathlib import Path


class ClimbingConfig:

    def __init__(self, path_to_config):

        self.data = []

        with open(path_to_config) as f:
            self.data = json.load(f)

        self.assets = self.data["assets"]
        self.sim_steps_per_action = self.data["sim_steps_per_action"]
        self.holds = self.data["holds"]
        self.stance_path = self.data["stance_path"]
        self.climber = self.data["climber"]
        self.surface = self.data["surface"]
        self.plane = self.data["ground_plane"]

        for key in self.holds:
            self.holds[key]['asset_data'] = self.assets[self.holds[key]['asset']]

        self.surface['asset_data'] = self.assets[self.surface['asset']]
        self.plane['asset_data'] = self.assets[self.plane['asset']]

