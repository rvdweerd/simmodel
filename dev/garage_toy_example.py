import gym
from stable_baselines3.common.env_checker import check_env
from modules.rl.environments import GraphWorld
import simdata_utils as su
from rl_custom_worlds import GetCustomWorld

# Select graph world
env=GetCustomWorld('Manhattan3x3_PauseFreezeWorld', make_reflexive=False, state_repr='et', state_enc='tensor')
#env=GetCustomWorld('Manhattan3x3_PauseDynamicWorld', make_reflexive=False, state_repr='et', state_enc='nodes')

# Convert to Garage environment
import click
import numpy as np
from garage.envs import GymEnv
from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.envs import normalize
from garage.tf.optimizers import (ConjugateGradientOptimizer,
                                  FiniteDifferenceHVP)
from garage.tf.policies import CategoricalMLPPolicy, CategoricalLSTMPolicy
from garage.trainer import TFTrainer
#from garage.envs import PointEnv
from garage.sampler import LocalSampler
import keras
#import torch 

class NpWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        obs = np.array(observation).astype(np.int64)
        return obs
env.render_fileprefix='./images/garage/testrun'
#env = NpWrapper(env)
genv = GymEnv(env, max_episode_length=10)
#s=genv.reset()
#genv.step(0)
#genv.render()

# Use LSTM-based recurrent policy
@click.command()
@click.option('--seed', default=1)
@click.option('--n_epochs', default=10000)
@click.option('--batch_size', default=64)
@click.option('--plot', default=False)

@wrap_experiment
def trpo_recurrent(ctxt, seed, n_epochs, batch_size, plot):
    """Train TRPO with a recurrent policy.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        n_epochs (int): Number of epochs for training.
        seed (int): Used to seed the random number generator to produce
            determinism.
        batch_size (int): Batch size used for training.
        plot (bool): Whether to plot or not.

    """
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        env = normalize(genv)#GymEnv('CartPole-v1', max_episode_length=100)

        policy = CategoricalLSTMPolicy(name='policy', env_spec=env.spec)#, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = LocalSampler(
            agents=policy,
            envs=env,
            max_episode_length=env.spec.max_episode_length,
            is_tf_worker=True)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    sampler=sampler,
                    discount=0.99,
                    max_kl_step=0.01,
                    optimizer=ConjugateGradientOptimizer,
                    optimizer_args=dict(hvp_approach=FiniteDifferenceHVP(
                        base_eps=1e-5)))

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=batch_size, plot=plot)

trpo_recurrent()