import gymnasium as gym
import numpy as np
import sys
import os 
sys.path.append('../')
#from stable_baselines3.common import BaseAlgorithm
from roadmap import Roadmap
import argparse
from gym_envs.factory import CarLikeFactory


class ReachabilityRoadmap(Roadmap):

    def __init__(self, config, env: gym.Env, controller):

        self.controller = controller
        self.e = config["epsilon"]
        self.max_steps_c = config["max_rollout_steps_in_construction"]
        super().__init__(env,config)

    #build roadmap with termination parameter max_ssa - after max_ssa consecutive failed additions, terminate
    #forces termination after term_override steps
    def build(self, max_ssa, term_override = 1e9):
        steps = 0
        stepsSinceAdd = 0
        self.components = {}
        self.component_count = 0
        self.condensation_graph_edges = {}
        while(steps < term_override and stepsSinceAdd < max_ssa):
            sample = self.env.observation_space.sample()
            self.nodes[self.node_count] = sample

            ### START: Get indices of arrivals/departures
            arrivals = {}
            departures = {}
            #managing component handling.
            c_arr = {}
            c_dep = {}
            c_merge = {}
            if len(self.components)>0:
                print(self.components)
                for c_indx , c in self.components.items():
                    a = False
                    d = False
                    for indx in c:
                        if(self._controller_rollout(self.node_count,indx,self.max_steps_c)):
                            a = True
                            arrivals.add(indx)
                        if (self._controller_rollout(indx,self.node_count,self.max_steps_c)):
                            d = True
                            departures.add(indx)
                    if a and d:
                        c_merge.add(c_indx)
                    elif a:
                        c_arr.add(c_indx)
                    elif d:
                        c_dep.add(c_indx)
            ### END get indices

            if(len(c_merge) == 1 and len(c_arr)+len(c_dep) <= 1):
                stepsSinceAdd += 1
                pass
            else:
                # Add node as new component
                sample_indx = self.node_count
                self.node_count += 1
                self.nodes[sample_indx] = sample
                self.edges[sample_indx] = []
                new_c_indx = self.component_count
                self.component_count += 1
                self.components[new_c_indx] = {sample_indx}
                self.condensation_graph_edges[new_c_indx] = []
                
                # add base graph edges
                for a in arrivals:
                    self.edges[a].add(sample_indx)
                for d in departures:
                    self.edges[sample_indx].add(d)

                # add condensation graph edges
                for c_a in c_arr:
                    self.condensation_graph_edges[c_a].add( new_c_indx)
                for c_d in c_dep:
                    self.condensation_graph_edges[new_c_indx].add(c_d)
                for c_m in c_merge:
                    self.merge_components(c_m,new_c_indx, fcc_cleanup=False)
                if(len(c_merge)>0):
                    self.fully_connected_cleanup()
            steps += 1
        return
    
    def merge_components(self, c1, c2, fcc_cleanup = True):
        #merge nodes
        for n in self.components[c2]:
            self.components[c1].add(n)
        #merge edges
        for e in self.condensation_graph_edges:
            if c2 in e:
                for i in range(len(e)):
                    if e[i] == c2:
                        e[i] = c1
        #remove c2
        self.components.pop(c2)
        #cleanup fcc loops created by merge
        if(fcc_cleanup):
            self.fully_connected_cleanup()
        
    
    def fully_connected_cleanup(self):
        merges = {}
        for c1 in self.components:
            merges[c1] = []
            for c2 in self.components:
                if c1 != c2 and c2 not in merges.keys:
                    if( self.check_connectivity(c1,c2) and self.check_connectivity(c2,c1)):
                        merges[c1].push(c2)
        for base, mergeset in merges:
            for merge in mergeset:
                self.merge_components(base, merge, fcc_cleanup=False)
        

    #perform dfs on component graph to find path from index d to a
    def check_connectivity(self, a, d):
        s = []
        visited = {}
        s.push(d)
        visited.add(d)
        while len(s) > 0:
            c = s.peek()
            s.pop
            if c == a:
                return True
            if len(self.condensation_graph_edges[c]) > 0:
                for e in self.condensation_graph_edges[c]:
                    if e not in visited:
                        s.push(e)
                        visited.add(e)
        return False

    def _controller_rollout(self, start_indx, targ_indx, max_steps):
        obs, info = self.env.reset(options = {"start":self.nodes[start_indx],"goal":self.nodes[targ_indx]})
        traj = [obs['observation']]
        goal = obs['desired_goal']
        start = obs['achieved_goal']
        plan = []
        timestep = 0
        done = False
        while (not done) and timestep < max_steps:
            action, _ = self.controller.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = self.env.step(action)
            traj.append(info['traj'])
            #TODO: fix done check.
            timestep += 1
            done = done or trunc

        return done
    

argparser = argparse.ArgumentParser()

argparser.add_argument('--no_velocity_goals', default=False, action='store_true')
argparser.add_argument('--train_config', type=str, default="analytical_mushr_zero_goal")
argparser.add_argument('--alg', choices=['PPO', "HER_SAC", "BangBang"], type=str, default="HER_SAC")
argparser.add_argument('--model_path', type=str, default=os.path.dirname(__file__).removesuffix("Roadmap")+'trained_models/latest/best/best_model')
argparser.add_argument('--plan_file', type=str, default='plan.txt')
argparser.add_argument('--traj_file', type=str, default='simulated_traj.txt')
argparser.add_argument('--plot', action='store_true')
argparser.add_argument('--max_steps', type=int, default=1e10)


dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    args = argparser.parse_args()

    exp_config_fpath = os.path.join(os.path.dirname(__file__).removesuffix("/Roadmap"), 'configs', f'{args.train_config}.txt')

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


    roadmap = ReachabilityRoadmap(config=config, env=env, controller=model)
    roadmap.build(10)