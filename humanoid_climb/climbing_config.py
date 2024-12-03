import json
from pathlib import Path


class ClimbingConfig:
    HOLDS_PREDEFINED = 'predefined'

    TRANSITIONS_PREDEFINED = 'predefined'

    def __init__(self, sim_config_path, climb_config_path):

        sim_config = None
        climb_config = None

        # read general sim config
        with open(sim_config_path) as f:
            sim_config = json.load(f)

        self.assets = sim_config['assets']
        self.sim_steps_per_action = sim_config['sim_steps_per_action']
        self.climber = sim_config['climber']
        self.surface = sim_config['surface']
        self.surface['asset_data'] = self.assets[self.surface['asset']]
        self.plane = sim_config['ground_plane']
        self.plane['asset_data'] = self.assets[self.plane['asset']]

        # check if climb config given (i.e. hold configuration and stance path)
        with open(climb_config_path) as f:
            climb_config = json.load(f)

        self.hold_definition = climb_config['hold_definition']
        if self.hold_definition == self.HOLDS_PREDEFINED:
            self.holds = climb_config['holds']
            for key in self.holds:
                self.holds[key]['asset_data'] = self.assets[self.holds[key]['asset']]
        else:
            self.holds = []

        self.transition_definition = climb_config['transition_definition']
        if self.transition_definition == self.TRANSITIONS_PREDEFINED:
            self.stance_path = climb_config['stance_path']

        self.init_states_fn = None
        if 'init_states_fn' in climb_config.keys():
            self.init_states_fn = climb_config['init_states_fn']

