import json


class ClimbingConfig:
    HOLDS_PREDEFINED = 'predefined'
    HOLDS_DYNAMIC = 'dynamic'

    TRANSITIONS_PREDEFINED = 'predefined'
    TRANSITIONS_DYNAMIC = 'dynamic'

    DEFAULT_HOLD_ASSET_DATA = {
        "path": "./humanoid_climb/assets/target.xml",
        "type": "URDF"
    }

    def __init__(self, sim_config_path, climb_config_path):
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

        # read climb config (hold configuration and stance path, etc.)
        with open(climb_config_path) as f:
            climb_config = json.load(f)

        self.holds = None
        self.hold_parameters = None
        self.hold_definition = climb_config['hold_definition']
        if self.hold_definition == self.HOLDS_PREDEFINED:
            self.holds = climb_config['holds']
            for key in self.holds:
                self.holds[key]['asset_data'] = self.assets[self.holds[key]['asset']]
        elif self.hold_definition == self.HOLDS_DYNAMIC:
            self.hold_parameters = climb_config['hold_parameters']
        else:
            raise ValueError(f'unknown hold_definition in config: {self.hold_definition}')

        self.stance_path = None
        self.transition_parameters = None
        self.transition_definition = climb_config['transition_definition']
        if self.transition_definition == self.TRANSITIONS_PREDEFINED:
            self.stance_path = climb_config['stance_path']
        elif self.transition_definition == self.TRANSITIONS_DYNAMIC:
            self.transition_parameters = climb_config['transition_parameters']
        else:
            raise ValueError(f'unknown transition_definition in config: {self.transition_definition}')

        self.init_states_fn = None
        if 'init_states_fn' in climb_config.keys():
            self.init_states_fn = climb_config['init_states_fn']
