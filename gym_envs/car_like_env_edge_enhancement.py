import numpy as np
import os
import gymnasium as gym

from utils import norm_angle_pi, LOSS_MODE_TYPES


class CarLikeEnvEE(gym.Env):
    L = 0.6

    def __init__(self, config, has_velocity_goals: bool = False, return_full_trajectory=False, return_dict_obs=False):
        self.has_velocity_goals = has_velocity_goals
        self.return_full_trajectory = return_full_trajectory
        self.return_dict_obs = return_dict_obs

        # Copy values from Config
        self.max_steps = config['max_steps']
        self.prop_steps = config['prop_steps']
        self.env_limit = config['env_limit']
        self.goal_limit = config['goal_limit']
        self.dt = config['dt']
        self.velocity_limits = config['velocity_limits']
        self.steering_angle_limits = config['steering_angle_limits']
        self.acceleration_limits = config['acceleration_limits'] 
        
        self.root_start = config['start']
        self.start_size = config['epsilon']
        self.root_goal = config['goal']

        # Load Obstacles
        # raise NotImplementedError()
        # with open(yaml_prefix + args.file, 'r') as stream:
        #     try:
        #         self.env_params = yaml.load(stream, Loader=yaml.FullLoader)
        #     except yaml.YAMLError as exc:
        #         print(exc)

        # try:
        #     for geom in self.env_params['environment']['geometries']:
        #         self.collision_geometry = geom['collision_geometry']
        #         name = geom['name']
        #         config = geom['config']
        #         if collision_geometry['type'] == 'box':
        #             print("Box geometry found with name ", name)
        #             dims = collision_geometry['dims']
        #             pos = config['position']
        #             quat = config['orientation']
        # except yaml.YAMLError as exc:
        #         print(exc)


        
        self._process_loss_mode(config)
        self._process_distance_threshold(config)
        self._process_goal_limit()
        if self.has_velocity_goals: self._process_start_velocity(config)

        self.obs_dims = 4  # x,y,theta,v
        self.goal_dims = 4 if has_velocity_goals else 3  # x,y,theta,v with velocity goals else x, y, theta
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))

        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dims,))

        if not self.return_dict_obs:
            self.observation_space = observation_space
        else:
            self.observation_space = gym.spaces.Dict({
                "observation": observation_space,
                "achieved_goal": gym.spaces.Box(low=self.achieved_goal_limit[:, 0], high=self.achieved_goal_limit[:, 1]),
                "desired_goal": gym.spaces.Box(low=self.goal_limit[:, 0], high=self.goal_limit[:, 1], shape=(self.goal_dims,))
            })

    def _get_obs(self):
        if not self.return_dict_obs:
            return np.float32(self.state)
        return {
            "observation": np.float32(self.state),
            "achieved_goal": np.float32(self.state[:self.goal_dims]),
            "desired_goal": np.float32(self.goal)
        }
    
    def _process_goal_limit(self):
        """
        Sets the value of self.goal_limit and self.achieved_goal_limit based on the value of config['goal_limit']
        Replaces all occurrences of 'low_vel' and 'high_vel' with the actual velocity limits and all occurrences of 'env_limit' with the actual env_limit
        """
        self.goal_limit = np.array(self.goal_limit)

        if self.has_velocity_goals:
            # Replace all occurrences of 'low_vel' and 'high_vel' with the actual velocity limits
            self.goal_limit[self.goal_limit == 'low_vel'] = self.velocity_limits[0]
            self.goal_limit[self.goal_limit == 'high_vel'] = self.velocity_limits[1]
        else:
            # Remove the velocity limits from the goal limit
            self.goal_limit = self.goal_limit[:3]

        # Replace all occurrences of 'env_limit' with the actual env_limit
        self.goal_limit[self.goal_limit == '-env_limit'] = -self.env_limit
        self.goal_limit[self.goal_limit == 'env_limit'] = self.env_limit

        # Replace all occurrences of 'pi' with the actual pi
        self.goal_limit[self.goal_limit == '-pi'] = -np.pi
        self.goal_limit[self.goal_limit == 'pi'] = np.pi

        self.goal_limit = self.goal_limit.astype(np.float32)

        self.achieved_goal_limit = np.copy(self.goal_limit)

        if self.has_velocity_goals:
            self.achieved_goal_limit[3, 0] = self.velocity_limits[0]

    def _process_distance_threshold(self, config):
        """
        Sets the value of self.distance_threshold based on the value of config['distance_threshold']
        The first distance threshold is always present
            If combine_all_dist is True, then this is the only distance threshold
            If combine_pos_angle_dist is True, then this is the distance threshold for position and angle combined
            If both are False, then this is the distance threshold for position
        """

        distance_threshold_config = config['distance_threshold']


        distance_threshold = [distance_threshold_config[0]]

        if not self.combine_all_dist:
            if not self.combine_pos_angle_dist:
                # Angle's distance threshold is appended
                distance_threshold.append(distance_threshold_config[1])

            if self.has_velocity_goals:
                # Velocity's distance threshold is appended
                distance_threshold.append(distance_threshold_config[2])

        self.distance_threshold = np.array(distance_threshold).reshape((-1, 1))

    def _process_start_velocity(self, config):
        """
        Sets the value of self.start_velocity based on the value of config['start_vel']
        """

        start_velocity = config['start_vel']

        if start_velocity == 'high_vel':
            self.start_velocity = self.velocity_limits[1]
        elif start_velocity == 'low_vel':
            self.start_velocity = self.velocity_limits[0]
        else:
            self.start_velocity = start_velocity

    def _process_loss_mode(self, config):
        """
        Sets the value of self.combine_all_dist and self.combine_pos_angle_dist based on the value of config['loss_mode']
        """

        if config['loss_mode'] == LOSS_MODE_TYPES['COMBINE_ALL_DIST']:
            self.combine_all_dist = True    
            self.combine_pos_angle_dist = False
        elif config['loss_mode'] == LOSS_MODE_TYPES['COMBINE_POS_AND_ANGLE']:
            self.combine_all_dist = False
            self.combine_pos_angle_dist = True
        else:
            self.combine_all_dist = False
            self.combine_pos_angle_dist = False
    
    def rand_perturb_point(self, pt, radius, dims):
        vec = np.random.rand(self.dims)
        vec /= np.linalg.norm(vec, axis=0)
        vec *= radius
        vec += pt
        return vec

    def reset(self, goal=None):
        """
        complete override of start and goal states
        """

        self.steps = 0
        self.state = np.zeros((self.obs_dims,))

        nstart = np.array(self.rand_perturb_point(self.root_start, self.start_size, self.goal_dims))
        assert ngoal.shape == (self.goal_dims, )
        self.state = nstart

        ngoal = np.array(self.root_goal)
        assert ngoal.shape == (self.goal_dims, )
        self.goal = ngoal


        if self.has_velocity_goals:
            if self.start_velocity == 'random':
                self.state[3] = np.random.uniform(self.velocity_limits[0],
                                                  self.velocity_limits[1])
            else:
                self.state[3] = self.start_velocity
        return self._get_obs()

    def _propagate_dynamics(self, state, action):
        print(state)
        # state: [x, y, theta, v]
        # action: [a, phi]
        derivative = np.zeros((self.obs_dims,))
        derivative[0] = state[3]*np.cos(state[2])
        derivative[1] = state[3]*np.sin(state[2])
        derivative[2] = state[3]*np.tan(action[0])/self.L
        derivative[3] = action[1]

        state += derivative*self.dt
        # Clip the angle
        state[2] = norm_angle_pi(state[2])
        # Clip the velocity
        state[3] = np.clip(state[3], self.velocity_limits[0],
                           self.velocity_limits[1])
        return np.copy(state)

    def goal_distance(self, goal_a, goal_b, loss_mode=None):
        """
        If self.combine_all_dist is True, then returns the distance between euclidian goal_a and goal_b which combines the loss of position, angle and velocity
        If self.combine_pos_angle_dist is True, then returns an array with
            euclidian distance between the vector [position, angle] of goal_a and goal_b
            absolute difference between the velocity of goal_a and goal_b
        If both are False, then returns an array with
            euclidian distance between the positions of goal_a and goal_b
            absolute difference between the angle of goal_a and goal_b
            absolute difference between the velocity of goal_a and goal_b

        :param goal_a:
        :param goal_b:
        :param loss_mode:
        :return: np.array of goal distance with shape depending on the loss mode
        """
        assert goal_a.shape == goal_b.shape

        if loss_mode is None:
            combine_all_dist = self.combine_all_dist
            combine_pos_angle_dist = self.combine_pos_angle_dist
        else:
            if loss_mode == LOSS_MODE_TYPES['COMBINE_ALL_DIST']:
                combine_all_dist = True
                combine_pos_angle_dist = False
            elif loss_mode == LOSS_MODE_TYPES['COMBINE_POS_AND_ANGLE']:
                combine_all_dist = False
                combine_pos_angle_dist = True
            elif loss_mode == LOSS_MODE_TYPES['INDIVIDUAL']:
                combine_all_dist = False
                combine_pos_angle_dist = False

        if len(goal_a.shape) == 1:
            goal_a = goal_a[None,]
            goal_b = goal_b[None,]

        dist = []

        # Euclidian Distance between goal_a and goal_b which combines the loss of position, angle and velocity
        if combine_all_dist:
            dist.append(
                np.linalg.norm(goal_a[:, :4] - goal_b[:, :4], axis=-1),
            )
        else:
            # Euclidian distance between the vector [position, angle] of goal_a and goal_b
            if combine_pos_angle_dist:
                dist.append(
                    np.linalg.norm(goal_a[:, :3] - goal_b[:, :3], axis=-1),
                )
            else:
                # Euclidian distance between the positions of goal_a and goal_b
                dist.append(
                    np.linalg.norm(goal_a[:, :2] - goal_b[:, :2], axis=-1),
                )

                # Absolute difference between the angle of goal_a and goal_b
                dist.append(
                    np.abs(norm_angle_pi(goal_a[:, 2] - goal_b[:, 2])),
                )

            # Absolute difference between the velocity of goal_a and goal_b
            if self.has_velocity_goals:
                dist.append(
                    np.abs(goal_a[:, 3] - goal_b[:, 3])
                )

        dist = np.stack(dist)

        return dist if goal_a.shape[0] != 1 else dist.reshape((-1,))

    def _terminal(self, s, g):
        dist = self.goal_distance(s, g)
        distance_threshold = self.distance_threshold.reshape((-1,)) if len(dist.shape) == 1 else self.distance_threshold.reshape((-1, 1))
        return np.all(dist < distance_threshold,
                      axis=0)

    def compute_reward(self, achieved_goal, desired_goal, info):
        dist = self.goal_distance(achieved_goal, desired_goal)
        distance_threshold = self.distance_threshold.reshape((-1,)) if len(dist.shape) == 1 else self.distance_threshold.reshape((-1, 1))
        return - (np.any(
            dist > distance_threshold,
            axis=0
        )).astype(np.float32)
    
    def get_applied_action(self, action):
        applied_action = np.copy(action)

        # Scale action[0] from [-1,1] to [-np.pi/3,np.pi/3]
        # applied_action[0] = (action[0] + 1) / 2 * (self.steering_angle_limits[1] - self.steering_angle_limits[0]) + self.steering_angle_limits[0]

        # Scale action[1] from [-1,1] to [acceleration_limits[0],acceleration_limits[1]]
        applied_action[1] = (action[1] + 1) / 2 * (self.acceleration_limits[1] - self.acceleration_limits[0]) + self.acceleration_limits[0]

        return applied_action

    def step(self, action):
        self.steps += 1

        applied_action = self.get_applied_action(action)

        current_traj = []
        for _ in range(self.prop_steps):
            self.state = self._propagate_dynamics(self.state, applied_action)
            if self.return_full_trajectory:
                current_traj.append(self._get_obs()["achieved_goal"])

        obs = self._get_obs()

        if not self.return_dict_obs:
            achieved_goal = obs[:4]
        else:
            achieved_goal = obs['achieved_goal']

        info = {
            "is_success": self._terminal(achieved_goal, self.goal),
            "traj": np.array(current_traj)
        }

        done = self.steps >= self.max_steps or info["is_success"]
        reward = self.compute_reward(achieved_goal, self.goal, info)

        return obs, reward, done, info


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "config/car_like.txt")) as f:
        env_config = eval(f.read())
    carlike = CarLikeEnv(env_config)
