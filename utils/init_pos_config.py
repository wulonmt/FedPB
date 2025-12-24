import numpy as np
import gymnasium as gym

INIT_POS = {
    "MountainCarFixPos-v0": [
        {"init_x": 0, "x_limit": 0.15},
        {"init_x": -0.3, "x_limit": 0.15},
        {"init_x": -0.6, "x_limit": 0.15},
        {"init_x": 0.3, "x_limit": 0.15},
        {"init_x": -1, "x_limit": -1}
    ],
    "MountainCarFixPos-v1": [
        {"init_x": 0, "x_limit": 0.15},
        {"init_x": -0.3, "x_limit": 0.15},
        {"init_x": -0.6, "x_limit": 0.15},
        {"init_x": 0.3, "x_limit": 0.15},
        {"init_x": -1, "x_limit": -1}
    ],
    "PendulumFixPos-v0": [
        {"init_theta": np.pi*3/4, "init_thetadot": 1},
        {"init_theta": -np.pi*3/4, "init_thetadot": 1},
        {"init_theta": np.pi/2, "init_thetadot": 1},
        {"init_theta": -np.pi/2, "init_thetadot": 1},
        {"init_theta": -1, "init_thetadot": -1}
    ],
    "PendulumFixPos-v1": [
        {"init_theta": np.pi*3/4, "init_thetadot": 1},
        {"init_theta": -np.pi*3/4, "init_thetadot": 1},
        {"init_theta": np.pi/2, "init_thetadot": 1},
        {"init_theta": -np.pi/2, "init_thetadot": 1},
        {"init_theta": -1, "init_thetadot": -1}
    ],
    "CartPoleSwingUpFixInitState-v1": [
        {"init_x": 2, "init_angle": np.pi/2},
        {"init_x": -2, "init_angle": np.pi/2},
        {"init_x": 2, "init_angle": -np.pi/2},
        {"init_x": -2, "init_angle": -np.pi/2},
        {"init_x": 0, "init_angle": np.pi}
    ],
    "CartPoleSwingUpFixInitState-v2": [
        {"init_x": 2, "init_angle": np.pi/2},
        {"init_x": -2, "init_angle": np.pi/2},
        {"init_x": 2, "init_angle": -np.pi/2},
        {"init_x": -2, "init_angle": -np.pi/2},
        {"init_x": 0, "init_angle": np.pi}
    ],
    "HopperFixLength-v0": [
        {'thigh_scale': 1.0, 'leg_scale': 1.0}, # default
        {'thigh_scale': 1.2, 'leg_scale': 1.2}, # train high
        {'thigh_scale': 0.8, 'leg_scale': 0.8}, # train low
        {'thigh_scale': 1.1, 'leg_scale': 1.1}, # test in high
        {'thigh_scale': 1.1, 'leg_scale': 0.9}, # test in chaos
        {'thigh_scale': 1.4, 'leg_scale': 1.4}, # test out high
        {'thigh_scale': 0.5, 'leg_scale': 0.5}, # test out low
    ],
    "HalfCheetahFixLength-v0": [
        {'bthigh_scale': 1.0, 'fthigh_scale': 1.0}, # default
        {'bthigh_scale': 1.25, 'fthigh_scale': 1.25}, # train high
        {'bthigh_scale': 0.75, 'fthigh_scale': 0.75}, # train low
        {'bthigh_scale': 1.1, 'fthigh_scale': 1.1}, # test in high
        {'bthigh_scale': 1.2, 'fthigh_scale': 0.8}, # test in chaos
        {'bthigh_scale': 1.5, 'fthigh_scale': 1.5}, # test out high
        {'bthigh_scale': 0.5, 'fthigh_scale': 0.5} # test out los
    ],
    "CrowdedHighway-v0": [
        {"density": 2, "count": 50},
        {"density": 2, "count": 100},
        {"density": 0.5, "count": 50},
        {"density": 0.5, "count": 100},
        {"density": -1, "count": -1},
    ],
    "CrowdedHighway-v1": [
        {"density": 2, "count": 50},
        {"density": 2, "count": 100},
        {"density": 0.5, "count": 50},
        {"density": 0.5, "count": 100},
        {"density": -1, "count": -1},
    ],
    "CarRacingFixSeed-v0": [
        {"index": 0},
        {"index": 1},
        {"index": 2},
        {"index": 3},
        {"index": 4}
    ],
    "CartPoleSwingUpActionScale-v1": [
        {"scale": 1},
        {"scale": 0.7},
        {"scale": 0.5},
        {"scale": 0.3},
        {"scale": 0.1}
    ],
    "MountainCarActionScale-v1": [
        {"scale": 1},
        {"scale": 0.85},
        {"scale": 0.7},
        {"scale": 0.55},
        {"scale": 0.6}
    ],
    "PendulumActionScale-v1": [
        {"scale": 1},
        {"scale": 0.8},
        {"scale": 0.55},
        {"scale": 0.3},
        {"scale": 0.05}
    ],
    "CartPoleSwingUpV1WithAdjustablePole-v0": [
        {"pole_length": 0.6},   # default
        {"pole_length": 0.7},   # training high
        {"pole_length": 0.5},   # training low
        {"pole_length": 0.65},  # test in range high
        {"pole_length": 0.55},  # test in range low
        {"pole_length": 0.8},   # test out range high
        {"pole_length": 0.4},   # test out range low
    ],
    "Pendulum-v1": [
        {"g": 10.0},  # Default
        {"g": 12.5},  # Training High
        {"g": 7.5},   # Training Low
        {"g": 11.0},  # Test In-Range
        {"g": 9.0},   # Test In-Range
        {"g": 14.0},  # Test Out-Range High
        {"g": 5.0},   # Test Out-Range Low
    ]
}

def get_init_pos(env_name, index):
    """
    獲取指定環境和索引的初始位置參數
    
    :param env_name: 環境名稱
    :param index: 初始位置索引
    :return: 初始位置參數字典
    """
    pos_len = len(INIT_POS[env_name])
    return INIT_POS[env_name][min(index, pos_len - 1)]

def get_init_list(env_name):
    """
    獲取指定環境列表
    
    :param env_name: 環境名稱
    :return: 環境初始位置列表
    """
    return INIT_POS[env_name]

def get_param_names(env_name):
    """
    獲取指定環境的參數名稱列表
    
    :param env_name: 環境名稱
    :return: 參數名稱列表
    """
    return list(INIT_POS[env_name][0].keys())

def is_valid_env(env_name):
    """
    檢查環境名稱是否有效
    
    :param env_name: 環境名稱
    :return: 如果環境名稱有效則返回 True，否則返回 False
    """
    gym_envs = gym.envs.registry
    return env_name in INIT_POS or env_name in gym_envs

def is_costum_env(env_name):
    """
    檢查環境名稱是否為自定義環境
    
    :param env_name: 環境名稱
    :return: 如果環境名稱為自定義環境則返回 True，否則返回 False
    """
    return env_name in INIT_POS

def get_available_envs():
    """
    獲取所有可用的環境名稱
    
    :return: 可用環境名稱的列表
    """
    return list(INIT_POS.keys())

def assert_alarm(env_name):
    assert is_valid_env(env_name), f"Only environments {', '.join(get_available_envs())} are available"