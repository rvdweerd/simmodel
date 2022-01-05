#import argparse
#import gym
#import simdata_utils as su
from stable_baselines3.common.env_checker import check_env
#from modules.rl.environments import GraphWorld
from modules.rl.rl_policy import EpsilonGreedyPolicySB3_PPO
from Phase1_hyperparameters import GetHyperParams_SB3PPO
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.rl_utils import EvaluatePolicy, print_parameters, GetFullCoverageSample, NpWrapper
#from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DQN
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

world_name='Manhattan5x5_FixedEscapeInit'
state_repr='etUte0U0'
exp_rootdir='./results/PPO/'+world_name+'/'+state_repr+'/'
env_train=GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc='tensors')
env_train = NpWrapper(env_train)
hp=GetHyperParams_SB3PPO(world_name)
eval_deterministic= hp['eval_determ'][env_train.state_representation]
sample_multiplier = hp['sampling_m'][env_train.state_representation]
model = PPO.load(exp_rootdir+'Model_best')

world_name='Manhattan5x5_VariableEscapeInit'
env_test=GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc='tensors')
env_test = NpWrapper(env_test)
policy=EpsilonGreedyPolicySB3_PPO(env_test, model, deterministic=eval_deterministic)
lengths, returns, captures = EvaluatePolicy(env_test, policy, env_test.world_pool*sample_multiplier, print_runs=False, save_plots=False, logdir=exp_rootdir+'/gen')
plotlist = GetFullCoverageSample(returns, env_test.world_pool*sample_multiplier, bins=10, n=10)
EvaluatePolicy(env_test, policy, plotlist, print_runs=True, save_plots=True, logdir=exp_rootdir+'/gen')
