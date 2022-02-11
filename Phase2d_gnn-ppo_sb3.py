import argparse
import gym
import simdata_utils as su
from stable_baselines3.common.env_checker import check_env
from modules.rl.environments import GraphWorld
from modules.rl.rl_custom_worlds import GetCustomWorld
from modules.rl.rl_utils import EvaluatePolicy, print_parameters, GetFullCoverageSample, NpWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DQN
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
