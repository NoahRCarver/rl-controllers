import gymnasium as gym
import numpy as np
import sys
import os 
sys.path.append('../')
#from stable_baselines3.common import BaseAlgorithm
from roadmap import Roadmap
import argparse
from gym_envs.car_like_env import CarLikeEnv
from gym_envs.factory import CarLikeFactory


class ReachabilityRoadmap(Roadmap):

    def __init__(self, config, env: gym.Env, controller, ctrl_env: gym.Env):

        self.controller = controller
        self.e = config["epsilon"]
        self.max_steps_c = config["max_rollout_steps_in_construction"]
        if ctrl_env != None:
            self.ctrl_env = ctrl_env
            self.is_ctrl_env = True
            #TODO: assert action spaces are same
        else:
            self.ctrl_env = env
            self.is_ctrl_env = False

        print("env: ",env.observation_space)
        print("ctrl_env: ", ctrl_env.observation_space)
        self.uenv = env.unwrapped
        super().__init__(env,config)

    #build roadmap with termination parameter max_ssa - after max_ssa consecutive failed additions, terminate
    #forces termination after term_override steps
    def build(self, max_ssa, term_override = 1e9):

        steps = 0
        stepsSinceAdd = 0
        self.components = {}
        self.component_count = 0
        self.condensation_graph_edges = {}
        num_UCS = 0

    
        while(steps < term_override and stepsSinceAdd < max_ssa-num_UCS):
            sample = np.random.uniform(self.uenv.start_limit[:, 0], self.uenv.start_limit[:, 1], size=(self.uenv.obs_dims,))
            while self.env.unwrapped.pt_collision_check(sample[:2]):
                sample = np.random.uniform(self.uenv.start_limit[:, 0], self.uenv.start_limit[:, 1], size=(self.uenv.obs_dims,))
                
            #print("sample: ", sample)

            self.nodes[self.node_count] = sample

            ### START: Get indices of arrivals/departures
            arrivals = set()
            departures = set()
            #managing component handling.
            c_arr = set()
            c_dep = set()
            c_merge = set()
            if len(self.components)>0:
                for c_indx , c in self.components.items():
                    a = False
                    d = False
                    for indx in c:
                        dep_success,_ = self._controller_rollout(self.node_count,indx,self.max_steps_c)
                        arr_success,_ = self._controller_rollout(indx,self.node_count,self.max_steps_c)
                        if(arr_success):
                            a = True
                            arrivals.add(indx)
                        if (dep_success):
                            d = True
                            departures.add(indx)
                    if a and d:
                        c_merge.add(c_indx)
                    elif a:
                        c_arr.add(c_indx)
                    elif d:
                        c_dep.add(c_indx)
            ### END get indices
            thief_caught = False
            
            if( len(c_merge) == 1 ):
               #todo:check if all arrivals and departures are in cmerge's a's and d's
               pass
            elif(len(c_merge) == 0):
                #todo: check if all a's arrive at all d's on comp_graph

                #If A = nullset, its a guard.
                #if D = nullset, its a thief and is caught
                pass
            raise NotImplementedError()
            if(thief_caught): # todo: better comp_graph update check
                stepsSinceAdd += 1
            else:
                stepsSinceAdd = 0
                # Add node as new component
                sample_indx = self.node_count
                self.node_count += 1
                self.nodes[sample_indx] = sample
                self.edges[sample_indx] = []
                new_c_indx = self.component_count
                self.component_count += 1
                self.components[new_c_indx] = {sample_indx}
                self.condensation_graph_edges[new_c_indx] = set()
                
                # add base graph edges
                for a in arrivals:
                    self.edges[a].append(sample_indx)
                for d in departures:
                    self.edges[sample_indx].append(d)

                # add condensation graph edges
                for c_a in c_arr:
                    self.condensation_graph_edges[c_a].add( new_c_indx)
                for c_d in c_dep:
                    self.condensation_graph_edges[new_c_indx].add(c_d)
                for c_m in c_merge:
                    self.merge_components(new_c_indx,c_m, fcc_cleanup=False)
                if(len(c_merge)>0):
                    self.fully_connected_cleanup()
            steps += 1
            num_UCS = len([x for x in self.components.keys() if len(self.condensation_graph_edges[x])== 0])
            print("[update] step: ", steps, ", ssa: ", stepsSinceAdd, ", num_ucs: ", num_UCS)

        print("nodes: ", self.nodes)
        print("edges: ", self.edges)
        print("components: ", self.components)
        return
    
    def save_roadmap(self, roadmap_outdir):


        os.makedirs(os.path.join(roadmap_outdir,"trajs"), exist_ok=True)

        with open(os.path.join(roadmap_outdir,"nodes.txt"),'w+') as node_f:
            for index,pt in self.nodes.items():
                node_f.write(f"{index}, {','.join(pt.astype(str))}\n")

        with open(os.path.join(roadmap_outdir,"components.txt"),'w+') as comp_f:
            for index,comp in self.components.items():
                comp_f.write(f"{index}, {','.join(str(e) for e in comp)}\n")

        edge_number = 0
        with open(os.path.join(roadmap_outdir,"edges.txt"),'w+') as edge_f:
            for start_index,end_indices in self.edges.items():
                for end_index in end_indices:
                    edge_f.write(f"{edge_number}, {start_index}, {end_index}\n")

                    #make traj file
                    success, traj = self._controller_rollout(start_index, end_index,self.max_steps_c)
                    if not success:
                        print("SOMETHING IS VERY VERY WRONG")
                    with open(os.path.join(roadmap_outdir,"trajs", f"traj_{edge_number}.txt"),'w+') as traj_f:
                        for pt in traj:
                            traj_f.write(f"{','.join(pt.astype(str))}\n")
                    edge_number += 1
    
    def merge_components(self, c1, c2, fcc_cleanup = True):
        print("merge components ",c1," and ", c2)
        #print(self.condensation_graph_edges)
        #print(self.components)
        #merge nodes
        for n in self.components[c2]:
            self.components[c1].add(n)
        #merge edges
        
        for s,e in self.condensation_graph_edges.items():
            if (s == c1 and c2 in e):
                e.remove(c2)
            if(s != c1 and s!= c2):
                if c2 in e:
                    e.remove(c2)
                    e.add(c1)
        #remove c2
        self.components.pop(c2)
        self.condensation_graph_edges.pop(c2)
        #cleanup fcc loops created by merge
        if(fcc_cleanup):
            self.fully_connected_cleanup()
          
    def fully_connected_cleanup(self):
        
        merges = {}
        clean_trigger = False
        for c1 in self.components:
            merges[c1] = []
            for c2 in self.components:
                if c1 != c2 and c2 not in merges.keys():
                    if( self.check_connectivity(c1,c2) and self.check_connectivity(c2,c1)):
                        merges[c1].append(c2)
                        clean_trigger=True
        if clean_trigger:
            print("Cleanup")
            for base in sorted(merges.keys()):
                if base in self.components.keys():
                    for merge in merges[base]:
                        if merge in self.components.keys():
                            self.merge_components( base, merge, fcc_cleanup=False)

    #perform dfs on component graph to find path from index d to a
    def check_connectivity(self, a, d):
        s = []
        visited = set()
        s.append(d)
        visited.add(d)
        while len(s) > 0:
            c = s.pop()
            if c == a:
                return True
            if len(self.condensation_graph_edges[c]) > 0:
                for e in self.condensation_graph_edges[c]:
                    if e not in visited:
                        s.append(e)
                        visited.add(e)
        return False

    def _controller_rollout(self, start_indx, targ_indx, max_steps):
        #print("q:\tstart: ", self.nodes[start_indx], ";\n\tgoal: ", self.nodes[targ_indx][:-1] )
        obs, info = self.env.reset(options = {"start":self.nodes[start_indx],"goal":self.nodes[targ_indx][:-1]})
        traj = [obs['observation']]
        goal = obs['desired_goal']
        start = obs['observation']

        plan = []
        timestep = 0
        done = False
        trunc = False
        while not(done or trunc) and timestep < max_steps:
            ctrl_obs = obs
            if self.is_ctrl_env:
                ctrl_obs = self._transform_obs(obs) #transform observation to controller obs

            action, _ = self.controller.predict(ctrl_obs, deterministic=True)
            obs, reward, done, trunc, info = self.env.step(action)
            traj.extend(info['traj'])
            timestep += 1
        return done, traj
    
    def _transform_obs(self, obs, method = "zero_goal"):
        new_obs = {}
        x, y, theta, v, phi = np.copy(obs['observation'])
        gx, gy, gt, gv = obs['desired_goal']
        if method == "zero_goal":
            c, s = np.cos(-gt), np.sin(-gt )
            x-=gx
            y-=gy
            tx,ty = c*x-s*y, s*x + c*y
            x, y = tx,ty
            theta = theta - gt
            new_obs['observation'] = [x, y, theta, v, phi]
            new_obs['achieved_goal'] = [0, 0, 0, v]
            new_obs['desired_goal'] = [0, 0, 0, gv]
        return new_obs
     
     

argparser = argparse.ArgumentParser()

argparser.add_argument('--no_velocity_goals', default=False, action='store_true')
argparser.add_argument('--train_config', type=str, default="analytical_mushr")
argparser.add_argument('--alg', choices=['PPO', "HER_SAC", "BangBang"], type=str, default="HER_SAC")
argparser.add_argument('--model_path', type=str, default=os.path.dirname(__file__).removesuffix("Roadmap")+'trained_models/latest/best/best_model')
argparser.add_argument('--max_steps', type=int, default=1e10)
argparser.add_argument('--output', type=str, default="roadmap_files/latest")


dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    args = argparser.parse_args()

    exp_config_fpath = os.path.join(os.path.dirname(__file__).removesuffix("/Roadmap"), 'configs', f'{args.train_config}.txt')
    print(exp_config_fpath)
    with open(exp_config_fpath, 'r') as f:
        config = eval(f.read())
    print(config)
    
    if(config["model_uses_alt_env"]):
        model_env_config_fpath = os.path.join(os.path.dirname(__file__).removesuffix("/Roadmap"), 'configs', f'{config["model_env_config"]}.txt')
        print(model_env_config_fpath)
        with open(model_env_config_fpath, 'r') as f2:
            model_env_config = eval(f2.read())
        print(model_env_config)

        

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

    
    if(config["model_uses_alt_env"]):
        
        env_factory_2 = CarLikeFactory(exp_config=model_env_config, return_full_trajectory=True)

        env_factory_2.register_environments_with_position_orientation_velocity_zero_goals()

        model_env = gym.make(model_env_config['env_name'])
    else:
        model_env = env

    if args.alg == 'HER_SAC':
        from stable_baselines3 import SAC
        model = SAC.load(args.model_path, env=model_env)


    roadmap = ReachabilityRoadmap(config=config, env=env, controller=model, ctrl_env=model_env)
    roadmap.build(10, term_override= 30)

    
    os.makedirs(os.path.join(args.output), exist_ok = True)
    roadmap_dir = os.path.join(os.path.dirname(__file__),args.output)

    roadmap.save_roadmap(roadmap_dir)
    #output files



    #testing