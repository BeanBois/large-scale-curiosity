# import os

# def guess_available_gpus(n_gpus=None):
#     if n_gpus is not None:
#         print('a')
#         return list(range(n_gpus))
#     if 'CUDA_VISIBLE_DEVICES' in os.environ:
#         cuda_visible_divices = os.environ['CUDA_VISIBLE_DEVICES']
#         cuda_visible_divices = cuda_visible_divices.split(',')
#         print('b')
#         return [int(n) for n in cuda_visible_divices]
#     nvidia_dir = '/proc/driver/nvidia/gpus/'
#     if os.path.exists(nvidia_dir):
#         n_gpus = len(os.listdir(nvidia_dir))
#         print('c')
#         return list(range(n_gpus))
#     raise Exception("Couldn't guess the available gpus on this machine")


# print(guess_available_gpus())

# try:
#     from OpenGL import GLU
# except:
#     print("no OpenGL.GLU")
import functools
import os.path as osp
from functools import partial

import gym
import rlbench.gym
import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor
# from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
from mpi4py import MPI
# print(guess_available_gpus())

from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dynamics import Dynamics, UNet
from utils import random_agent_ob_mean_std
import enum

class CameraView(enum.Enum):
    Left_shoulder = "left_shoulder_rgb"
    Right_shoulder = "right_shoulder_rgb"
    Wrist = "wrist_rgb"
    Front = "front_rgb"
    
def start_experiment(**args):
    print('set make env')
    make_env = partial(make_env_all_params, add_monitor=True, args=args)
    print('set trainer')
    
    trainer = Trainer(make_env=make_env,
                      num_timesteps=args['num_timesteps'], hps=args,
                      envs_per_process=args['envs_per_process'])
    print('set log')
    log, tf_sess = get_experiment_environment(**args)
    print('got experiment env')
    with log, tf_sess:
        logdir = logger.get_dir()
        print("results will be saved to ", logdir)
        trainer.train()


class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process):
        self.make_env = make_env
        self.hps = hps #hyper-parameters
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        print('start setting env_vars')
        self._set_env_vars()
        print('finish setting env_vars')

        #cleared till here

        self.policy = CnnPolicy(
            scope='pol',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            hidsize=512,
            feat_dim=512,
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nl=tf.nn.leaky_relu)

        self.feature_extractor = {"none": FeatureExtractor,
                                  "idf": InverseDynamics,
                                  "vaesph": partial(VAE, spherical_obs=True),
                                  "vaenonsph": partial(VAE, spherical_obs=False),
                                  "pix2pix": JustPixels}[hps['feat_learning']]
        self.feature_extractor = self.feature_extractor(policy=self.policy,
                                                        features_shared_with_policy=False,
                                                        feat_dim=512,
                                                        layernormalize=hps['layernorm'])

        self.dynamics = Dynamics if hps['feat_learning'] != 'pix2pix' else UNet
        self.dynamics = self.dynamics(auxiliary_task=self.feature_extractor,
                                      predict_from_pixels=hps['dyn_from_pixels'],
                                      feat_dim=512)

        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            dynamics=self.dynamics
        )

        self.agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['aux']
        self.agent.to_report['dyn_loss'] = tf.reduce_mean(self.dynamics.loss)
        self.agent.total_loss += self.agent.to_report['dyn_loss']
        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])

    #TODO: ob_space and ac_space needs processing. This processing is done in rlbench/gym/rlbench_env.py
    def _set_env_vars(self, camera_view=CameraView.Wrist):
        print('make beta env')
        #First create 'beta' environment with make_env_all_params() method
        env = self.make_env(0, add_monitor=False) #at this point no gpus avail thus cant run
        print('set ob_space and ac_space')
        self.ob_space, self.ac_space = env.observation_space[camera_view], env.action_space
        print(self.ob_space)
        print(self.ac_space)
        
        print('set ob_mean and ob_std')
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        print('del beta env')
        # if args["env_kind"] == "robo_env":
        #     #for robo env lets create 1 env first
        #     self.envs = [functools.partial(self.make_env, i) for i in range(1)] 
        # else: 
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

    def train(self):
        print('start training')
        self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamics=self.dynamics)
        while True:
            print('step')
            info = self.agent.step()
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
            if self.agent.rollout.stats['tcount'] > self.num_timesteps:
                break

        self.agent.stop_interaction()

#TODO: Amend function to accommodate for robogym env, and also try to see how how function like 'make_robo_XXX' works
#RESOLVED
def make_env_all_params(rank, add_monitor, args): #rank is 0, 
    env = gym.make(args['env'])

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
    return env


def get_experiment_environment(**args):
    from utils import setup_mpi_gpus, setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed
    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    setup_mpi_gpus()

    logger_context = logger.scoped_configure(dir=None,
                                             format_strs=['stdout', 'log',
                                                          'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else ['log'])
    tf_context = setup_tensorflow_session()
    return logger_context, tf_context


def add_environments_params(parser):
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4',
                        type=str)
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=4500, type=int)
    parser.add_argument('--env_kind', type=str, default="atari")
    parser.add_argument('--noop_max', type=int, default=30)


def add_optimization_params(parser):
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--norm_adv', type=int, default=1)
    parser.add_argument('--norm_rew', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ent_coeff', type=float, default=0.001)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--num_timesteps', type=int, default=int(1e8))


def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=128)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    parser.add_argument('--envs_per_process', type=int, default=128)
    parser.add_argument('--nlumps', type=int, default=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser)
    add_optimization_params(parser)
    add_rollout_params(parser)

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=0.)
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--layernorm', type=int, default=0)
    parser.add_argument('--feat_learning', type=str, default="none", 
                        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"])

    args = parser.parse_args()
    print('starting experiment')
    start_experiment(**args.__dict__) 
 