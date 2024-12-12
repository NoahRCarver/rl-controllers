# Script to evaluate an already trained model.
import argparse
import json
import os
from functools import partial

from stable_baselines3 import HER, SAC

import matplotlib.pyplot as plt
import numpy as np
import torch
import gym

from utils import LOSS_MODE_TYPES, load_config, load_best_model, initialize_env, parallel_run

# Number of trials to average the evaluation are:
num_eval_runs = 30


def evaluate(model, env, num_episodes=100, verbose=False):
    all_episode_successes = []
    all_episode_lengths = []

    for i in range(num_episodes):
        done = False
        info = None
        obs = env.reset()
        ep_len = 0
        while not done:
            ep_len += 1
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
        all_episode_successes.append(info['is_success'])
        all_episode_lengths.append(ep_len)

    success_rate = 1.0 * sum(all_episode_successes) / num_episodes
    mean_ep_length = 1.0 * sum(all_episode_lengths) / num_episodes
    if verbose: print("Success Rate: ", success_rate)
    return success_rate, mean_ep_length


def get_data_for_plots(model, env, num_episodes=100):
    all_achieved_goals = []
    all_desired_goals = []
    all_ep_lens = []
    success_indices = []
    failure_indices = []

    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        ep_len = 0
        while not done:
            ep_len += 1
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
        all_ep_lens.append(ep_len)
        all_achieved_goals.append(np.copy(obs['achieved_goal']))
        all_desired_goals.append(np.copy(obs['desired_goal']))

        if info['is_success']:
            success_indices.append(i)
        else:
            failure_indices.append(i)
    return np.vstack(all_achieved_goals), np.vstack(all_desired_goals), success_indices, failure_indices, np.array(
        all_ep_lens)


def plot_final_states(achieved, desired, indices_to_plot, plot_name, model_path, env_name, config):
    plt.figure(figsize=(8, 8))
    plt.xlim([-config['env_limit'] - 2, config['env_limit'] + 2])
    plt.ylim([-config['env_limit'] - 2, config['env_limit'] + 2])
    plt.scatter(achieved[indices_to_plot, 0], achieved[indices_to_plot, 1], label="Achieved goal")
    plt.scatter(desired[indices_to_plot, 0], desired[indices_to_plot, 1], label="Desired goal")
    # Plot the angle of achieved and desired goals.
    for i in indices_to_plot:
        min_vel_val = 0.5
        achieved_vel_sign = np.sign(achieved[i, 3])
        desired_vel_sign = np.sign(desired[i, 3])

        if config['no_velocity_goals']:
            achieved_vel = 0.5
            desired_vel = 0.5
        elif abs(achieved[i, 3]) > abs(desired[i, 3]):
            denom = desired[i, 3] if desired[i, 3] != 0 else 1e-6
            if abs(achieved[i, 3] / denom) > 5:
                min_vel_val = 0.2
            achieved_vel = min_vel_val * achieved[i, 3] / denom
            desired_vel = min_vel_val
        else:
            denom = achieved[i, 3] if achieved[i, 3] != 0 else 1e-6
            if abs(desired[i, 3] / denom) > 5:
                min_vel_val = 0.2
            desired_vel = min_vel_val * desired[i, 3] / denom
            achieved_vel = min_vel_val

        achieved_vel = np.clip(abs(achieved_vel), 0, 5) * achieved_vel_sign
        desired_vel = np.clip(abs(desired_vel), 0, 5) * desired_vel_sign

        # Achieved goal
        plt.plot(
            (
                achieved[i, 0],
                achieved[i, 0] + achieved_vel * np.cos(achieved[i, 2])
            ),
            (
                achieved[i, 1],
                achieved[i, 1] + achieved_vel * np.sin(achieved[i, 2])
            ),
            color='black',
        )

        # Desired goal
        plt.plot(
            (
                desired[i, 0],
                desired[i, 0] + desired_vel * np.cos(desired[i, 2])
            ),
            (
                desired[i, 1],
                desired[i, 1] + desired_vel * np.sin(desired[i, 2])
            ),
            color='black',
        )

        # Draw line from achieved to desired goal.
        plt.plot((achieved[i, 0], desired[i, 0]), (achieved[i, 1], desired[i, 1]), color='grey', linestyle='--')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.title(env_name)
    plt.savefig(os.path.join(model_path, plot_name))


def plot_goal_limit_analysis(goal_limit_analysis, model_name, model_path, loss_mode):
    goal_limits = goal_limit_analysis.keys()

    success_rate_means = []
    error_means = []
    ep_len_means = []
    episode_len_stddevs = []

    for goal_limit in goal_limits:
        analysis = goal_limit_analysis[goal_limit]
        success_rate_means.append(analysis['success_rate_mean'] * 100)
        error_means.append(analysis['error_mean'])
        ep_len_means.append(analysis['episode_len_mean'])
        episode_len_stddevs.append(analysis['episode_len_stddev'])

    error_means = np.array(error_means)

    x_axis = np.arange(len(goal_limits))

    plt.figure(figsize=(12, 12))
    plt.ylim((40, 100))
    plt.bar(goal_limits, success_rate_means, width=0.4)
    plt.xticks(x_axis, goal_limits)
    plt.xlabel("Position Goal Limits")
    plt.ylabel("Success Rate")
    plt.suptitle('Analytical MuSHR Goal Limit Success Analysis - Success Rate')
    plt.title(f'Cost Mode - {loss_mode} | Model - {model_name}')
    plt.savefig(os.path.join(model_path, 'goal_limit_analysis_success.png'))
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.bar(x_axis - 0.2, error_means[:, 0], 0.2, label='Position')
    plt.bar(x_axis, error_means[:, 1], 0.2, label='Orientation')
    plt.bar(x_axis + 0.2, error_means[:, 2], 0.2, label='Velocity')
    plt.xticks(x_axis, goal_limits)
    plt.legend()
    plt.xlabel("Position Goal Limits")
    plt.ylabel("Distance from Goal")
    plt.suptitle('Analytical MuSHR Goal Limit Error Analysis - Error')
    plt.title(f'Cost Mode - {loss_mode} | Model - {model_name}')
    plt.savefig(os.path.join(model_path, 'goal_limit_analysis_error.png'))
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.errorbar(goal_limits, ep_len_means, episode_len_stddevs, fmt='o', capsize=3)
    plt.xticks(x_axis, goal_limits)
    plt.xlabel("Position Goal Limits")
    plt.ylabel("Trajectory Length (Mean and Deviation)")
    plt.suptitle('Analytical MuSHR Goal Limit Success Analysis - Trajectory Length')
    plt.title(f'Cost Mode - {loss_mode} | Model - {model_name}')
    plt.savefig(os.path.join(model_path, 'goal_limit_analysis_ep_len.png'))


def main(args, model_name=None):
    try:
        if args.not_verbose:
            def tqdm(x):
                return x
        else:
            from tqdm import tqdm

        model_path = os.path.join(os.path.dirname(__file__), f'trained_models/{model_name or args.model_name}')

        print('Evaluating:\n', model_path)

        config = load_config(model_path)

        env_name = config['env_name']
        alg = config['alg']

        env = initialize_env(config)
        model = load_best_model(model_path, env, alg)

        evaluations = dict()

        evals = np.load(os.path.join(model_path, "logs/evaluations.npz"))
        ep_lengths = evals["ep_lengths"]
        tsteps = evals["timesteps"]
        successes = evals["successes"]

        plt.figure(figsize=(8, 8))
        plt.grid()
        plt.errorbar(tsteps / 1000, np.mean(ep_lengths, axis=1), yerr=np.std(ep_lengths, axis=1), label="SAC")
        plt.ylabel("Average Trajectory length")
        plt.xlabel("Environment steps (x1000)")
        plt.legend()
        plt.savefig(os.path.join(model_path, "training_trajlen.png"))

        plt.figure(figsize=(8, 8))
        plt.grid()
        plt.errorbar(tsteps / 1000, np.mean(successes, axis=1), yerr=np.std(successes, axis=1), label="SAC")
        plt.ylabel("Average success rate")
        plt.xlabel("Environment steps (x1000)")
        plt.legend()
        plt.savefig(os.path.join(model_path, "training_success.png"))

        eval_success_rates = np.zeros((num_eval_runs,))
        eval_ep_lengths = np.zeros((num_eval_runs,))

        for i in tqdm(range(num_eval_runs)):
            eval_success_rates[i], eval_ep_lengths[i] = evaluate(model, env, verbose=not args.not_verbose)

        achieved, desired, success_indices, failure_indices, ep_lens = get_data_for_plots(model, env)

        goal_errors = np.array([
            env.goal_distance(achieved[i, :], desired[i, :], loss_mode=LOSS_MODE_TYPES['INDIVIDUAL'])
            for i in range(achieved.shape[0])
        ])

        if not args.not_verbose:
            print("Success rate (mean): ", np.mean(eval_success_rates))
            print("Success rate (stddev): ", np.std(eval_success_rates))

            print("Episode Length (mean): ", np.mean(eval_ep_lengths))
            print("Episode Length (stddev): ", np.std(eval_ep_lengths))

            print("Error mean: ", goal_errors.mean(axis=0))
            print("Error stddev: ", goal_errors.std(axis=0), "\n")

            print("Success Error mean: ", goal_errors[success_indices].mean(axis=0))
            print("Success Error stddev: ", goal_errors[success_indices].std(axis=0), "\n")

            print("Failure Error mean: ", goal_errors[failure_indices].mean(axis=0))
            print("Failure Error stddev: ", goal_errors[failure_indices].std(axis=0), "\n")

        evaluations['success_rate_mean'] = np.mean(eval_success_rates)
        evaluations['success_rate_stddev'] = np.std(eval_success_rates)
        evaluations['episode_len_mean'] = np.mean(eval_ep_lengths)
        evaluations['episode_len_stddev'] = np.std(eval_ep_lengths)
        evaluations['error_mean'] = goal_errors.mean(axis=0).tolist()
        evaluations['error_stddev'] = goal_errors.std(axis=0).tolist()
        evaluations['success_error_mean'] = goal_errors[success_indices].mean(axis=0).tolist()
        evaluations['success_error_stddev'] = goal_errors[success_indices].std(axis=0).tolist()
        evaluations['failure_error_mean'] = goal_errors[failure_indices].mean(axis=0).tolist()
        evaluations['failure_error_stddev'] = goal_errors[failure_indices].std(axis=0).tolist()

        plot_final_states(achieved, desired, success_indices, "final_states_success.png", model_path, env_name, config)
        plot_final_states(achieved, desired, failure_indices, "final_states_failures.png", model_path, env_name, config)

        if args.goal_limit_analysis:
            env_limit = config['env_limit']
            orig_goal_limits = config['goal_limit']
            goal_limit_analysis_evals = dict()
            evaluations['goal_limit_analysis'] = goal_limit_analysis_evals

            for goal_limit in range(1, int(env_limit) + 1):
                curr_goal_limit_evals = dict()
                goal_limit_analysis_evals[goal_limit] = curr_goal_limit_evals

                new_config = config.copy()
                new_goal_limits = orig_goal_limits.copy()

                new_goal_limits[0] = [-goal_limit, goal_limit]
                new_goal_limits[1] = [-goal_limit, goal_limit]

                new_config['goal_limit'] = new_goal_limits

                new_env = initialize_env(new_config)

                eval_success_rates = np.zeros((num_eval_runs,))
                eval_ep_lengths = np.zeros((num_eval_runs,))

                for i in range(num_eval_runs):
                    eval_success_rates[i], eval_ep_lengths[i] = evaluate(model, new_env, verbose=not args.not_verbose)

                achieved, desired, success_indices, failure_indices, ep_lens = get_data_for_plots(model, new_env)

                goal_errors = np.array([
                    new_env.goal_distance(achieved[i, :], desired[i, :], loss_mode=LOSS_MODE_TYPES['INDIVIDUAL'])
                    for i in range(achieved.shape[0])
                ])

                if not args.not_verbose:
                    print(f"Goal Limit {goal_limit} Success rate (mean): ", np.mean(eval_success_rates))
                    print(f"Goal Limit {goal_limit} Success rate (stddev): ", np.std(eval_success_rates))

                    print(f"Goal Limit {goal_limit} Error mean: ", goal_errors.mean(axis=0))
                    print(f"Goal Limit {goal_limit} Error stddev: ", goal_errors.std(axis=0), "\n")

                    print(f"Goal Limit {goal_limit} Success Error mean: ", goal_errors[success_indices].mean(axis=0))
                    print(f"Goal Limit {goal_limit} Success Error stddev: ", goal_errors[success_indices].std(axis=0),
                          "\n")

                    print(f"Goal Limit {goal_limit} Failure Error mean: ", goal_errors[failure_indices].mean(axis=0))
                    print(f"Goal Limit {goal_limit} Failure Error stddev: ", goal_errors[failure_indices].std(axis=0),
                          "\n")

                curr_goal_limit_evals['success_rate_mean'] = np.mean(eval_success_rates)
                curr_goal_limit_evals['success_rate_stddev'] = np.std(eval_success_rates)
                curr_goal_limit_evals['episode_len_mean'] = np.mean(eval_ep_lengths)
                curr_goal_limit_evals['episode_len_stddev'] = np.std(eval_ep_lengths)
                curr_goal_limit_evals['error_mean'] = goal_errors.mean(axis=0).tolist()
                curr_goal_limit_evals['error_stddev'] = goal_errors.std(axis=0).tolist()
                curr_goal_limit_evals['success_error_mean'] = goal_errors[success_indices].mean(axis=0).tolist()
                curr_goal_limit_evals['success_error_stddev'] = goal_errors[success_indices].std(axis=0).tolist()
                curr_goal_limit_evals['failure_error_mean'] = goal_errors[failure_indices].mean(axis=0).tolist()
                curr_goal_limit_evals['failure_error_stddev'] = goal_errors[failure_indices].std(axis=0).tolist()

                plot_goal_limit_analysis(evaluations['goal_limit_analysis'], args.model_name, model_path,
                                         config['loss_mode'])

        with open(os.path.join(model_path, 'evaluation_results.txt'), 'w') as f:
            f.write(json.dumps(evaluations, indent=2))
    except Exception as e:
        print(e)
        print("Evaluation failed for model: ", model_name or args.model_name)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--model_name', type=str)
    argParser.add_argument('--goal_limit_analysis', action='store_true', default=False)
    argParser.add_argument('--parallel', action='store_true', default=False)
    argParser.add_argument('--not_verbose', action='store_true', default=False)
    argParser.add_argument('-l', '--list', nargs='+')

    args = argParser.parse_args()

    if args.parallel:  # TODO: Implement parallel evaluation.
        raise NotImplementedError("Parallel evaluation is not implemented yet.")

    if args.parallel and not args.not_verbose:
        print("Parallel evaluation is not verbose. Setting not_verbose to True.")
        args.not_verbose = True

    if args.list and len(args.list) > 0:
        if not args.parallel:
            for model_name in args.list:
                main(args, model_name)
        else:
            partial_main = partial(main, args)
            parallel_run(partial_main, args.list, show_progress=True)

        exit()

    if args.model_name is None:
        print("Please provide a model name.")
        exit()

    main(args)
