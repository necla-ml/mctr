import math
import json
from collections import defaultdict

import numpy as np

def parse_camera_param(camera_config_file):
    """Parse camera param from yaml file."""
    camera_parameters = defaultdict(dict)
    camera_configs = json.load(open(camera_config_file))
    camera_ids = list()

    for camera_param in camera_configs['Cameras']:

        cam_id = camera_param['CameraId']
        camera_ids.append(cam_id)

        camera_parameters[cam_id]['Translation'] = np.asarray(
                camera_param['ExtrinsicParameters']['Translation'])[np.newaxis, :]
        camera_parameters[cam_id]['Rotation'] = np.asarray(
                camera_param['ExtrinsicParameters']['Rotation']).reshape((3, 3))

        camera_parameters[cam_id]['FInv'] = np.asarray([
                1 / camera_param['IntrinsicParameters']['Fx'],
                1 / camera_param['IntrinsicParameters']['Fy'], 1
            ])[np.newaxis, :]
        camera_parameters[cam_id]['C'] = np.asarray([
                camera_param['IntrinsicParameters']['Cx'],
                camera_param['IntrinsicParameters']['Cy'], 0
            ])[np.newaxis, :]

        discretization_factorX = 1.0 / (
            (camera_configs['Space']['MaxU'] - camera_configs['Space']['MinU']) / (math.floor(
                (camera_configs['Space']['MaxU'] - camera_configs['Space']['MinU']) /
                camera_configs['Space']['VoxelSizeInMM']) - 1))
        discretization_factorY = 1.0 / (
            (camera_configs['Space']['MaxV'] - camera_configs['Space']['MinV']) / (math.floor(
                (camera_configs['Space']['MaxV'] - camera_configs['Space']['MinV']) /
                camera_configs['Space']['VoxelSizeInMM']) - 1))
        camera_parameters['discretization_factor'] = np.asarray([discretization_factorX, discretization_factorY, 1])

        camera_parameters['min_volume'] = np.asarray([
            camera_configs['Space']['MinU'], camera_configs['Space']['MinV'],
            camera_configs['Space']['MinW']
        ])
    
    return camera_parameters