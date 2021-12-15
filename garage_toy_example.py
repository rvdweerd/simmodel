import gym
from stable_baselines3.common.env_checker import check_env
from environments import GraphWorld
import simdata_utils as su
from rl_custom_worlds import GetCustomWorld

# Select graph world
env=GetCustomWorld('Manhattan3x3_PauseFreezeWorld', make_reflexive=False, state_repr='et', state_enc='nodes')
#env=GetCustomWorld('Manhattan3x3_PauseDynamicWorld', make_reflexive=False, state_repr='et', state_enc='nodes')

# Convert to Garage environment
from garage.envs import GymEnv
import numpy as np
from garage import wrap_experiment
from garage.envs import PointEnv
from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer
import torch 

class NpWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        obs = np.array(observation).astype(np.float)
        return obs
env = NpWrapper(env)
genv = GymEnv(env, max_episode_length=10)
genv.reset()
genv.render('human')


@wrap_experiment
def trpo_point(ctxt=None, seed=1):
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        env = normalize(genv)

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

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
                    max_kl_step=0.01)

        trainer.setup(algo, env)
        trainer.train(n_epochs=100, batch_size=4000)


#trpo_point()