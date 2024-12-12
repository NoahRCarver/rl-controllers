import json, re

import numpy as np
from datetime import datetime
from uuid import uuid4
from gym_envs.factory import CarLikeFactory
from utils import initialize_env, load_config, load_best_model

import gymnasium as gym
import os
import argparse

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv


np.set_printoptions(suppress=True)


def load_train_config(train_config, no_velocity_goals):
    # Set the experiment configuration file path
    exp_config_fpath = os.path.join(os.path.dirname(__file__), 'configs', f'{train_config}.txt')

    with open(exp_config_fpath) as f:
        # Read the configuration file
        config = eval(f.read())

    load_model_path = None

    if "load_model_name" in config:
        load_model_path = os.path.join(os.path.dirname(__file__), 'trained_models', config["load_model_name"])
        loaded_config = load_config(load_model_path)
        retrain_config = config
        config = {
            **loaded_config,
            **retrain_config,
            "loaded_config": loaded_config,
            "retrain_config": retrain_config
        }

        if config['no_velocity_goals'] != no_velocity_goals:
            print('WARNING: no_velocity_goals is different from the loaded model. Using the value from the loaded model.')

        if config['env_name'] != loaded_config['env_name']:
            print('WARNING: env_name is different from the loaded model. Using the value from the loaded model.')
            config['env_name'] = loaded_config['env_name']

        if config['alg'] != loaded_config['alg']:
            print('WARNING: alg is different from the loaded model. Using the value from the loaded model.')
            config['alg'] = loaded_config['alg']

        if config['loss_mode'] != loaded_config['loss_mode']:
            print('WARNING: loss_mode is different from the loaded model. Using the value from the retrain config')
    else:
        config['no_velocity_goals'] = no_velocity_goals

    return config, load_model_path

def initialize_model(config, env, load_model_path):
    if load_model_path is None:
        # Create the model based on the algorithm
        if config['alg'] == 'HER_SAC':
            from stable_baselines3 import HerReplayBuffer, SAC

            model = SAC(
                'MultiInputPolicy',
                env,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy='future',
                ),
                verbose=1,
                buffer_size=int(1e6),
                learning_starts=int(1e3),
                learning_rate=1e-3,
                gamma=0.95,
                batch_size=256
            )
        elif config['alg'] == 'PPO':
            from stable_baselines3 import PPO

            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                batch_size=256,
                gamma=0.95,
            )
        else:
            raise ValueError('Invalid algorithm: {}'.format(config['alg']))
    else:
        model = load_best_model(load_model_path, env, config['alg'], to_train=True)

    return model

def initialize_eval_env(config):
    if config['alg'] == 'HER_SAC':
        eval_env = DummyVecEnv([lambda: gym.make(config['env_name'])])
        # eval_env = ObsDictWrapper(eval_env)
    elif config['alg'] == 'PPO':
        eval_env = gym.make(config['env_name'])
    else:
        raise ValueError('Invalid algorithm: {}'.format(config['alg']))

    return eval_env

def train(args):
    config, load_model_path = load_train_config(args.train_config, args.no_velocity_goals)

    # Set the num steps from config file. If num_steps is not present, use 1e6
    num_steps = int(1e6 if 'num_steps' not in config else config['num_steps'])

    # Set the seed from config file. If seed is random, generate one
    seed = config['seed']

    if seed == 'random':
        seed = uuid4().int & (2 ** 32 - 1)
        config['seed'] = seed

    set_random_seed(seed, using_cuda=True)

    # Set the checkpoint path for saving the model and logs
    checkpt_path = os.path.dirname(__file__) + "/trained_models/{}_{}{}_{}/".format(
        config['env_name'],
        config['alg'],
        '_vel_goal' if not args.no_velocity_goals else '',
        datetime.now().strftime('%d_%m_%Y-%H_%M_%S_%f')[:-3]
    )

    print('Args', args)
    print('\nConfig', json.dumps(config, indent=2))
    print('\nSaving Model to:\n', checkpt_path, '\n')

    env = initialize_env(config)
    model = initialize_model(config, env, load_model_path)
    eval_env = initialize_eval_env(config)

    # Create the callbacks for saving the model and logging
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpt_path)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=100,
        eval_freq=int(1e4),
        best_model_save_path=checkpt_path + "best/",
        log_path=checkpt_path + "logs/",
        deterministic=True
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Create the checkpoint directory
    os.makedirs(checkpt_path)

    # Save the config file
    with open(checkpt_path + 'config.json', 'w') as f:
        f.write(json.dumps(config, indent=2))

    # Train the model
    model.learn(int(num_steps), callback=callback)

    # Save the model
    model.save(checkpt_path)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--no_velocity_goals', default=False, action='store_true')
    argParser.add_argument('--train_config', type=str, default="analytical_mushr_zero_goal")

    train(argParser.parse_args())
