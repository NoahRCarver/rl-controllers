import argparse
import json
import os
from os import path

import numpy as np
import torch
import gymnasium as gym

from gym_envs.factory import CarLikeFactory
from utils import load_config, load_best_model, initialize_env


def main(model_name):
    model_path = os.path.join(os.path.dirname(__file__), f'trained_models/{model_name}')

    print('Saving:\n', model_path)

    config = load_config(model_path)

    env_name = config['env_name']
    alg = config['alg']
    env = initialize_env(config)
    model = load_best_model(model_path, env, alg)

    policy = model.policy
    actor_model = torch.nn.Sequential(policy.actor.latent_pi, policy.actor.mu, torch.nn.Tanh())

    example = torch.rand(1,(env.unwrapped.observation_space_dims))

    policy.net_args['observation_space']
    actor_model.eval()
    actor_model.to('cpu')

    with torch.jit.optimized_execution(True):
        traced_script_module = torch.jit.trace(actor_model,example)

    save_policy = True

    if save_policy:
        print("model_path: ",model_path)
        save_path = os.path.join(model_path, f'{alg}{"_with_vel" if not config["no_velocity_goals"] else ""}_best.pt')
        print("Saving policy to: ", save_path)
        traced_script_module.save(save_path)


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--model_name', type=str, required=True)

    args = argParser.parse_args()

    main(model_name=args.model_name)
