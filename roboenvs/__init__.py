from asyncio import tasks
import gym
from gym.envs.registration import register

#TODO: ADAPT FOR RLBench

from RLBench import rlbench
import rlbench.backend.task as task
import rlbench.backend.task as task
import os
from rlbench.utils import name_to_task_class
from rlbench.gym.rlbench_env import RLBenchEnv

TASKS = [t for t in os.listdir(task.TASKS_PATH)
         if t != '__init__.py' and t.endswith('.py')]

for task_file in TASKS:
    task_name = task_file.split('.py')[0]
    task_class = name_to_task_class(task_name)
    register(
        id='%s-state-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'state'
        }
    )
    register(
        id='%s-vision-v0' % task_name,
        entry_point='rlbench.gym:RLBenchEnv',
        kwargs={
            'task_class': task_class,
            'observation_mode': 'vision'
        }
    )




#################

from .joint_pong import DiscretizeActionWrapper, MultiDiscreteToUsual

register(
    id='RoboschoolPong-v2',
    entry_point='.joint_pong:RoboschoolPongJoint',
    max_episode_steps=10000,
    tags={"pg_complexity": 20 * 1000000},
)

register(
    id='RoboschoolHockey-v1',
    entry_point='.joint_hockey:RoboschoolHockeyJoint',
    max_episode_steps=1000,
    tags={"pg_complexity": 20 * 1000000},
)


def make_robopong():
    return gym.make("RoboschoolPong-v2")


def make_robohockey():
    return gym.make("RoboschoolHockey-v1")
