from humanoid_climb.assets.robot_util import *


class Asset:
    def __init__(self, bullet_client, config_asset, config):
        self.id = None

        asset_name = config_asset['asset']
        asset_info = config['assets'][asset_name]

        if asset_info['type'] == 'URDF':
            self.id = bullet_client.loadURDF(fileName=asset_info['path'])
        elif asset_info['type'] == 'MJCF':
            self.id = bullet_client.loadMJCF(fileName=asset_info['path'])[0]

        bullet_client.resetBasePositionAndOrientation(self.id, config_asset['position'], config_asset['orientation'])

        self.parts, self.joints, self.ordered_joints, self.body = addToScene(bullet_client, [self.id])
