#!/usr/bin/env python
import gym
import safety_gym
from safety_gym.envs.engine import Engine

import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork

import time
import datetime
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')

from env_wrapper import env_wrapped

config = {
            'robot_base': 'xmls/car.xml',
            'task': 'push',
            'observe_goal_lidar': True,
            'observe_hazards': True,
            'observe_vases': True,
            'constrain_hazards': False,
            'lidar_max_dist': 3,
            'lidar_num_bins': 16,
            'hazards_num': 4,
            'vases_num': 4
        }



def main(config, robot, task, algo, seed, exp_name, cpu):

    # Verify experiment
    algo_list = ['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    assert algo in algo_list, "Invalid algo"

    # Hyperparameters
    exp_name = args.exp_name if not(args.exp_name == '') else '_'.join([algo, robot, task, timestamp]) 

    config['robot_base'] = 'xmls/' + args.robot + '.xml'
    config['task'] = args.task

    if robot=='doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 1e7
        steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 25

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = eval('safe_rl.'+algo)

    algo(env_fn=lambda: env_wrapped(config),
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs
         )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='doggo')
    parser.add_argument('--task', type=str, default='goal')
    parser.add_argument('--algo', type=str, default='ppo_lagrangian')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()

    main(config, args.robot, args.task, args.algo, args.seed, args.exp_name, args.cpu)
