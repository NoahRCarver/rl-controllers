import gymnasium as gym
import numpy as np
#from stable_baselines3.common import BaseAlgorithm


class Roadmap:
    def __init__(self, env, config):
        self.env = env
        self.nodes = {}
        self.edges = {}
        self.node_count = 0
        pass

    def build(self):
        return self.nodes, self.edges
    
    def query(self, start, goal):
        return None
