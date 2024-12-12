import os

from gym_envs.car_like_env import CarLikeEnv
from gym_envs.car_like_env_edge_enhancement import CarLikeEnvEE
from gymnasium.envs.registration import register

class CarLikeFactory:
    def __init__(self, exp_config, return_full_trajectory=False):
        self.exp_config = exp_config
        self.return_full_trajectory = return_full_trajectory
        self.return_dict_obs = (exp_config['alg'] == 'HER_SAC')

    def register_environments_for_edge_enhancement(self):
        register(
            id="CarLikeEnv-v0",
            entry_point=CarLikeEnvEE,
            kwargs={
                'config': self.exp_config,
                'return_full_trajectory': self.return_full_trajectory,
                'return_dict_obs': self.return_dict_obs,
            }
        )

    def register_environments_with_position_orientation_goals(self):
        register(
            id="CarLikeEnv-v0",
            entry_point=CarLikeEnv,
            kwargs={
                'config': self.exp_config,
                'return_full_trajectory': self.return_full_trajectory,
                'return_dict_obs': self.return_dict_obs,
            }
        )

    def register_environments_with_position_orientation_velocity_goals(self):
        register(
            id="CarLikeEnv_VG-v0",
            entry_point=CarLikeEnv,
            kwargs={
                'config': self.exp_config,
                'has_velocity_goals': True,
                'return_full_trajectory': self.return_full_trajectory,
                'return_dict_obs': self.return_dict_obs,
            }
        )
