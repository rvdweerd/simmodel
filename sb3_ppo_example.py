import gym
from stable_baselines3.common.env_checker import check_env
from environments import GraphWorld
from rl_policy import EpsilonGreedyPolicySB3_PPO
import simdata_utils as su
from rl_custom_worlds import GetCustomWorld
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_parameters(model):
    print(model)
    print('Policy model size:')
    print('------------------------------------------')
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print("{:44s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
    print("Total number of parameters: {}".format(total))
    print('------------------------------------------')
    assert total == sum(p.numel() for p in model.parameters() if p.requires_grad)


# Select graph world
#world_name='Manhattan3x3_PauseFreezeWorld'
#world_name='Manhattan3x3_PauseDynamicWorld'
#world_name='Manhattan5x5_FixedEscapeInit'
#world_name='Manhattan5x5_VariableEscapeInit'
#world_name='MetroU3_e17_FixedEscapeInit'
#world_name='MetroU3_e1t31_FixedEscapeInit'
world_name='MetroU3_e17t31_FixedEscapeInit'
env=GetCustomWorld(world_name, make_reflexive=True, state_repr='etUt', state_enc='tensor')

#su.SimulateInteractiveMode(env) # only works for state_enc='nodes'
#check_env(env)
# s=env.reset()
# env.render()
# print(s)

class NpWrapper(gym.ObservationWrapper):
    def _availableActionsInCurrentState(self):
        return None
    def observation(self, observation):
        obs = np.array(observation).astype(np.float64)
        return obs
env = NpWrapper(env)

#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DQN


# TO TRAIN
def train():
    policy_kwargs = dict(activation_fn = torch.nn.Tanh, net_arch = [dict(pi = [64, 64], vf = [64, 64])])
    policy_kwargs = dict(activation_fn = torch.nn.Tanh,net_arch = [dict(pi = [1028,1028,64], vf = [1028,1028,64])])
    for i in range(10):
        model = PPO('MlpPolicy', env, \
            #learning_rate=1e-4,\
            seed=i,\
            #clip_range=0.1,\    
            #max_grad_norm=0.1,\
            policy_kwargs = policy_kwargs, verbose=1, tensorboard_log="./sb3_ppo_tensorboard/")
        print_parameters(model.policy)
        model.learn(total_timesteps=80000)
        model.save('./models/sb3/ppo_'+world_name+'_run'+str(i))

def evaluate():
    #model = PPO.load('./models/sb3/ppo_'+world_name)
    #model = PPO.load('./results/sb3-PPO/MetroU3_e17_t31_etUt_T=20/ppo_MetroU3_e17t31_FixedEscapeInit')
    model = PPO.load('./results/sb3-PPO/MetroU3_e17_t0_etUt_T=20/ppo_MetroU3_e17t0_FixedEscapeInit_run9')
    policy=EpsilonGreedyPolicySB3_PPO(env, model, deterministic=True)
    from rl_utils import EvaluatePolicy
    EvaluatePolicy(env, policy, env.world_pool[1500:1501], print_runs=True, save_plots=True)    
    EvaluatePolicy(env, policy, env.world_pool, print_runs=False, save_plots=False)    

    # from stable_baselines3.common.evaluation import evaluate_policy
    # N_eval=10000
    #rewards, epi_lengths = evaluate_policy(model, env, n_eval_episodes=N_eval, deterministic=False, return_episode_rewards=True)
    #print(f"mean_reward={np.mean(rewards):.2f} +/- {np.std(rewards)}")
    #print(f"mean_lengths={np.mean(epi_lengths):.2f} +/- {np.std(epi_lengths)}")

evaluate()