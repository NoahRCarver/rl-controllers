import json, re, os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import numpy as np
import gymnasium as gym

def norm_angle_pi(angle):

    if angle.size > 1:
        while np.any(angle > np.pi):
            angle[angle > np.pi] -= 2*np.pi
        while np.any(angle < -np.pi):
            angle[angle < -np.pi] += 2*np.pi
        return np.clip(angle, -np.pi, np.pi)

    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return np.clip(angle, -np.pi, np.pi)


LOSS_MODE_TYPES = {
    'COMBINE_ALL_DIST': 'combine_all_dist',
    'COMBINE_POS_AND_ANGLE': 'combine_pos_angle_dist',
    'INDIVIDUAL': 'individual',
}


def load_config(model_path):
    if 'config.txt' in os.listdir(model_path):
        config_fpath = os.path.join(model_path, 'config.txt')
    elif 'config.json' in os.listdir(model_path):
        config_fpath = os.path.join(model_path, 'config.json')
    else:
        raise FileNotFoundError('No config file found in model directory')

    with open(config_fpath) as f:
        config = json.load(f)

    return config

def load_best_model(model_path, env, alg, to_train=False):
    model_to_load = os.path.join(model_path, 'best/best_model')
    if alg == 'HER_SAC':
        if to_train:
            from stable_baselines3 import HER
            model = HER.load(model_to_load, env=env, verbose=1)
        else:
            from stable_baselines3 import SAC
            model = SAC.load(model_to_load, env=env)
    elif alg == 'PPO':
        raise NotImplementedError()
        from stable_baselines3 import PPO

        model = PPO.load(model_to_load, env=env)
    else:
        raise NotImplementedError() 

    return model


def initialize_env(config):
    from gym_envs.factory import CarLikeFactory
    
    env_name = config['env_name']

    if config['geometry']:
        #handle loading geometry
        pass

    env_factory = CarLikeFactory(exp_config=config)

    if config['edge_enhancement']:
        env_factory.register_environments_for_edge_enhancement()

    else:
        if config['no_velocity_goals']:
            env_factory.register_environments_with_position_orientation_goals()
        else:
            env_factory.register_environments_with_position_orientation_velocity_goals()

    return gym.make(env_name)


def parallel_run(func, arg_list, show_progress=False):
    if not show_progress:
        with Pool(cpu_count() - 1) as p:
            return p.map(func, arg_list)
    results = []
    with Pool(min(cpu_count(), len(arg_list))) as p:
        for result in tqdm(p.imap_unordered(func, arg_list), total=len(arg_list)):
            results.append(result)
    return results