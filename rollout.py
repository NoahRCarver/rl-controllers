import os
import argparse
import math
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Wedge
import matplotlib.patches as mpatches
import os 
from gym_envs.factory import CarLikeFactory
from utils import LOSS_MODE_TYPES

argparser = argparse.ArgumentParser()
argparser.add_argument('--no_velocity_goals', default=False, action='store_true')
argparser.add_argument('--train_config', type=str, default="analytical_mushr")
argparser.add_argument('--alg', choices=['PPO', "HER_SAC", "BangBang"], type=str, default="HER_SAC")
argparser.add_argument('--model_path', type=str, default='trained_models/CL_VZG_HS_1/best/best_model')
argparser.add_argument('--plan_file', type=str, default='plan.txt')
argparser.add_argument('--traj_file', type=str, default='simulated_traj.txt')
argparser.add_argument('--plot', action='store_true')
argparser.add_argument('--max_steps', type=int, default=1e10)

dir_path = os.path.dirname(os.path.realpath(__file__))



def plotArrowAngle(x,y, theta, length = 1, offset = 0.5):

    x_head = x + offset*length*math.cos(theta)
    y_head = y + offset*length*math.sin(theta)
    x_tail = x - (1-offset)*length*math.cos(theta)
    y_tail = y - (1-offset)*length*math.sin(theta)

    return mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
                                 mutation_scale=10, color = "orange", alpha = 0.5)

def forward(args, obs):

    exp_config_fpath = os.path.join(os.path.dirname(__file__), 'configs', f'{args.train_config}.txt')
    with open(exp_config_fpath) as f:
        config = eval(f.read())
    config['alg'] = args.alg

    env_name = config['env_name']
    env_factory = CarLikeFactory(exp_config=config, return_full_trajectory=True)
    env_factory.register_environments_with_position_orientation_velocity_goals()
    env = gym.make(env_name)

    from stable_baselines3 import SAC
    model = SAC.load(args.model_path, env=env)

    action, _ = model.predict(obs, deterministic=True)

    return action

def rollout(args):
    if not os.path.exists('plans_and_trajs/'):
        os.makedirs('plans_and_trajs/')

    exp_config_fpath = os.path.join(os.path.dirname(__file__), 'configs', f'{args.train_config}.txt')

    with open(exp_config_fpath) as f:
        config = eval(f.read())

    #print('Args', args)
    #print('\nConfig', config)
    #print('\n')

    config['alg'] = args.alg

    env_name = config['env_name']
    

    env_factory = CarLikeFactory(exp_config=config, return_full_trajectory=True)

    if args.no_velocity_goals:
        env_factory.register_environments_with_position_orientation_goals()
    else:
        env_factory.register_environments_with_position_orientation_velocity_goals()

    env = gym.make(env_name)

    if args.alg == 'HER_SAC':
        from stable_baselines3 import SAC
        model = SAC.load(args.model_path, env=env)
    elif args.alg == 'PPO':
        from stable_baselines3 import PPO
        model = PPO.load(args.model_path, env=env)
    elif args.alg == 'BangBang':
        from bangBangController import BangBangController
        model = BangBangController()


    done = False
    # Enter the goal here.
    #{"start":[0., 0., 0., 0., 0.],"goal":[7.5, -7.5, np.pi/2, 0.]})
    obs, info = env.reset(options = {"start":args.start,"goal":args.goal})
    print(obs)
    traj = [obs['achieved_goal']]
    goal = obs['desired_goal']
    start = obs['achieved_goal']
    plan = []
    timestep = 0
    while not done and timestep < args.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        action_with_time = np.hstack([action, 1.0])
        plan.append(action_with_time)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        traj.append(info['traj'])

        timestep += 1
    plan = np.vstack(plan)
    traj = np.vstack(traj)

    if args.save_plan:
        np.savetxt(os.path.join(dir_path+'/plans_and_trajs/' + args.plan_file), plan, delimiter=',', fmt='%f')
    if args.save_traj:
        np.savetxt(os.path.join(dir_path+'/plans_and_trajs/' + args.traj_file), traj, delimiter=',', fmt='%f')

    d = config["distance_threshold"]
    if args.plot:
        plt.figure(figsize=(8, 8))
        plt.xlim(-env.unwrapped.env_limit, env.unwrapped.env_limit)
        plt.ylim(-env.unwrapped.env_limit, env.unwrapped.env_limit)
        plt.plot(traj[:, 0], traj[:, 1])
        circle = Circle((goal[0], goal[1]), d[0], color='green', alpha=0.5)
        arc = Wedge((goal[0],goal[1]), d[0], theta1=goal[2]-np.rad2deg(d[1]), theta2=goal[2]+np.rad2deg(d[1]), fill = True, color='blue', alpha=0.5 )
        plt.gca().add_patch(circle)
        plt.gca().add_patch(arc)
        arrow = plotArrowAngle(start[0],start[1],start[2])
        plt.gca().add_patch(arrow)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(np.array_str(start, precision=2)+" to " +np.array_str(goal, precision=2))
        plt.savefig(dir_path+"/plans_and_trajs/"+args.img_file)

    return env.unwrapped.terminal(traj[-1],goal), env.unwrapped.goal_distance(start,goal,LOSS_MODE_TYPES['COMBINE_ALL_DIST'])[0]

if __name__ == '__main__':

    args = argparser.parse_args()

    
    args.no_velocity_goals = False
    args.train_config = 'analytical_mushr_zero_goal'
    args.alg = 'HER_SAC'
    args.model_path = dir_path+'/trained_models/CL_VZG_HS_1/best/best_model'

    raw = "7.20575 4.63214 -1.4493 1 -0.5236 7.20575 4.63214 -1.4493 1 0 0 0 0 0.971262 -0.0978561"
    data = np.fromstring(raw, sep = " ")

    obs = {'observation': data[:5], 'achieved_goal': data[5:9], 'desired_goal': data[9:13]}
    out_data = forward(args, obs)
    print("input vector: ", data[:13])
    print("sb3 rl_controller: ", out_data)
    print("ml4kp learned_controller: ",data[13:])

    
    

    # quit


    # results = []
    # # nodes = np.array([ [-4.17407, -4.66024, 0.504611, 0],
    # #           [-1.95382,  6.38656,  2.22234, 0],
    # #           [ 4.91496,  2.53724, -2.96407, 0],
    # #           [ 2.51846,  5.69094, -2.30681, 0],
    # #           [-5.92729,  7.43364,  2.08309, 0],
    # #           [ 8.75710, -8.26489,  2.97140, 0],
    # #           [-1.56683,  6.61476,  1.07434, 0],
    # #           [-2.23892, -9.10369, -2.60551, 0],
    # #           [-1.83843, -7.67478, -2.24940, 0],
    # #           [-6.52077,  7.50903, -2.74895, 0],
    # #           [ 3.13479,  4.65402,  1.63715, 0],
    # #           [-8.29047,  4.20373, -3.09735, 0],
    # #           [-4.50943,  6.09398,-0.899594, 0],
    # #           [-3.51021, -0.149177, -1.81628, 0],
    # #           [-8.9859, 5.95172, 0.702144, 0]])
    # # N = len(nodes)
    # N = 10

    # for i in range(N):
    #     # for j in range(N):
    #     #     if i != j:
    #     args.no_velocity_goals = False
    #     args.train_config = 'analytical_mushr_zero_goal'
    #     args.alg = 'HER_SAC'
    #     args.model_path = dir_path+'/trained_models/CL_VZG_HS_1/best/best_model'
    #     args.plan_file = f'random_plan_{i}.txt'
    #     args.traj_file = f'random_traj_{i}.txt'
    #     args.img_file =  f'random_img_{i}.png'
    #     args.start = None #np.append(nodes[i],0.0)
    #     args.goal = [0., 0., 0., 0.] #nodes[j]
    #     args.plot = True
    #     args.save_plan = True
    #     args.save_traj = True
    #     args.max_steps = 150

    #     results.append(np.array(rollout(args)))


    # results = np.swapaxes(np.stack(results),0,1)
    # filter = results[0].astype(bool)
    # distances = results[1]
    # successes = distances[filter]
    # failures = distances[np.logical_not(filter)]

    # print("success:", successes.size, ", distances: mean =", np.mean(successes),", std = ", np.std(successes))
    # print("failure:", failures.size, ",  distances: mean =", np.mean(failures),", std = ", np.std(failures))

    # # Dict('achieved_goal': Box([ -7.  -7. -3.1415925 -0.2], [ 7.  7. 3.1415927 0.7], (4,), float32), 'desired_goal': Box([-7.        -7.        -3.1415925 -0.2      ], [7.        7.        3.1415927 0.7      ], (4,), float32), 'observation': Box(-inf, inf, (5,), float32)) != 
    # # Dict('achieved_goal': Box([-11. -11. -3.1415925 -0.2], [11. 11. 3.1415927 0.7], (4,), float32), 'desired_goal': Box([-11.        -11.         -3.1415925  -0.2      ], [11.        11.         3.1415927  0.7      ], (4,), float32), 'observation': Box(-inf, inf, (5,), float32))
