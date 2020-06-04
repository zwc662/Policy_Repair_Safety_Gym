#!/usr/bin/env python
import gym, gym.spaces
import safety_gym
from safety_gym.envs.engine import *

import numpy as np

class env_wrapped(Engine):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.edit_observation_space()


    def edit_observation_space(self):
        #if self.task == 'push':
        #    if self.observe_box_lidar:
        #        del self.obs_space_dict['box_lidar'] 
        #        self.obs_space_dict['box_dist'] = gym.spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        if self.observe_goal_lidar:
            del self.obs_space_dict['goal_lidar'] 
            self.obs_space_dict['goal_pos'] = gym.spaces.Box(0.0, 1.0, (2,), dtype=np.float32) 
        if self.observe_box_lidar:
            del obs['box_lidar'] 
            self.obs_space_dict['box_pos'] = gym.spaces.Box(0.0, 1.0, (2,), dtype = np.float32)  
        if self.task == 'circle' and selfle:
            del self.obs_space_dict['circle_lidar'] 
        if self.observe_hazards:
            del self.obs_space_dict['hazards_lidar'] 
            self.obs_space_dict['hazards_pos'] = gym.spaces.Box(0.0, 1.0, (self.config['hazards_num'], 2), dtype=np.float32) 
        if self.observe_vases:
            del self.obs_space_dict['vases_lidar'] 
            self.obs_space_dict['vases_pos'] = gym.spaces.Box(0.0, 1.0, (self.config['vases_num'], 2), dtype=np.float32) 
        if self.gremlins_num and self.observe_gremlins:
            del self.obs_space_dict['gremlins_lidar'] 
        if self.pillars_num and self.observe_pillars:
            del self.obs_space_dict['pillars_lidar'] 
            self.obs_space_dict['pillars_pos'] = gym.spaces.Box(0.0, 1.0, (self.pillars_num, 2), dtype=np.float32) 

    def obs(self):
        ''' Return the observation of our agent '''
        self.sim.forward()  # Needed to get sensordata correct
        obs = {}
        robot_pos = self.world.robot_pos()

        if self.observe_goal_dist:
            obs['goal_dist'] = np.array([np.exp(-self.dist_goal())])
        if self.observe_goal_comp:
            obs['goal_compass'] = self.obs_compass(self.goal_pos)
        if self.observe_goal_lidar:
            obs['goal_pos'] = np.asarray(self.goal_pos)[:2] - np.asarray(robot_pos)[:2]
        if self.task == 'push':
            box_pos = self.box_pos
            if self.observe_box_comp:
                obs['box_compass'] = self.obs_compass(box_pos)
            if self.observe_box_lidar:
                obs['box_pos'] = np.asarray(self.box_pos)[:2]- np.asarray(robot_pos)[:2]
        if self.task == 'circle' and self.observe_circle:
            obs['circle_lidar'] = self.obs_lidar([self.goal_pos], GROUP_CIRCLE)
        if self.observe_freejoint:
            joint_id = self.model.joint_name2id('robot')
            joint_qposadr = self.model.jnt_qposadr[joint_id]
            assert joint_qposadr == 0  # Needs to be the first entry in qpos
            obs['freejoint'] = self.data.qpos[:7]
        if self.observe_com:
            obs['com'] = self.world.robot_com()
        if self.observe_sensors:
            # Sensors which can be read directly, without processing
            for sensor in self.sensors_obs:  # Explicitly listed sensors
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.hinge_vel_names:
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.ballangvel_names:
                obs[sensor] = self.world.get_sensor(sensor)
            # Process angular position sensors
            if self.sensors_angle_components:
                for sensor in self.robot.hinge_pos_names:
                    theta = float(self.world.get_sensor(sensor))  # Ensure not 1D, 1-element array
                    obs[sensor] = np.array([np.sin(theta), np.cos(theta)])
                for sensor in self.robot.ballquat_names:
                    quat = self.world.get_sensor(sensor)
                    obs[sensor] = quat2mat(quat)
            else:  # Otherwise read sensors directly
                for sensor in self.robot.hinge_pos_names:
                    obs[sensor] = self.world.get_sensor(sensor)
                for sensor in self.robot.ballquat_names:
                    obs[sensor] = self.world.get_sensor(sensor)
        if self.observe_remaining:
            obs['remaining'] = np.array([self.steps / self.num_steps])
            assert 0.0 <= obs['remaining'][0] <= 1.0, 'bad remaining {}'.format(obs['remaining'])
        if self.walls_num and self.observe_walls:
            obs['walls_lidar'] = self.obs_lidar(self.walls_pos, GROUP_WALL)
        if self.observe_hazards:
            obs['hazards_pos'] = np.asarray(self.hazards_pos)[:, :2] - np.asarray(robot_pos)[:2]
        if self.observe_vases:
            obs['vases_pos'] = np.asarray(self.vases_pos)[:, :2] - np.asarray(robot_pos)[:2]
        if self.pillars_num and self.observe_pillars:
            obs['pillars_pos'] = np.asarray(self.pillars_pos)[:, :2] - np.asarray(robot_pos)[:2]
        if self.buttons_num and self.observe_buttons:
            # Buttons observation is zero while buttons are resetting
            if self.buttons_timer == 0:
                obs['buttons_lidar'] = self.obs_lidar(self.buttons_pos, GROUP_BUTTON)
            else:
                obs['buttons_lidar'] = np.zeros(self.lidar_num_bins)
        if self.observe_qpos:
            obs['qpos'] = self.data.qpos.copy()
        if self.observe_qvel:
            obs['qvel'] = self.data.qvel.copy()
        if self.observe_ctrl:
            obs['ctrl'] = self.data.ctrl.copy()
        if self.observe_vision:
            obs['vision'] = self.obs_vision()
        if self.observation_flatten:
            flat_obs = np.zeros(self.obs_flat_size)
            offset = 0
            for k in sorted(self.obs_space_dict.keys()):
                k_size = np.prod(obs[k].shape)
                flat_obs[offset:offset + k_size] = obs[k].flat
                offset += k_size
            obs = flat_obs
        assert self.observation_space.contains(obs), f'Bad obs {obs} {self.observation_space}'
        return obs

