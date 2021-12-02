from environments import GraphWorld
import numpy as np
import simdata_utils as su
import matplotlib.pyplot as plt
import torch
from dqn_utils import seed_everything, QNetwork, FastReplayMemory, EpsilonGreedyPolicyDQN, train, run_episodes
from rl_utils import EvaluatePolicy, SelectTrainset
from rl_policy import MinIndegreePolicy

import time
import os

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# Select comp device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Device = ',device)

# Select graph world
configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
conf=configs['Manhattan5']
#conf=configs['Manhattan11']
#conf=configs['CircGraph']
conf['direction_north']=False
fixed_init=conf['fixed_initial_positions']

# Define qnet 
dim_hidden=128

# Select hyperparameters
seed = 42  # This is not randomly chosen
batch_size      = 64
mem_size        = 16000
discount_factor = .9#1.#0.8
learn_rate      = 1e-3
num_episodes    = 8000#1500
eps_0           = 1.
eps_min         = 0.05
cutoff          = 0.7*num_episodes # lower plateau reached and maintained from this point onward
state_noise     = False

# Initialize
seed_everything(seed)
env = GraphWorld(conf, optimization_method='static', fixed_initial_positions=fixed_init, state_representation='etUte0U0', state_encoding='tensor')
# Select specific trainset, set0 has identical states with different rollouts, set1 has identical states with identical rollouts
init_pos_trainset_indices0, init_pos_trainset_indices1 = SelectTrainset(env, min_y_coord=env.sp.N-2, min_num_same_positions=env.sp.U, min_num_worlds=8, print_selection=False)
#env.world_pool = init_pos_trainset_indices1 # limit the training set to the selected entries
# Select full world pool
env.world_pool = env.all_worlds

dim_in = env.state_encoding_dim
dim_out = env.max_outdegree
memory = FastReplayMemory(mem_size,dim_in)
qnet = QNetwork(dim_in,dim_out,dim_hidden).to(device)
policy = EpsilonGreedyPolicyDQN(qnet, env, eps_0=eps_0, eps_min=eps_min, eps_cutoff=cutoff)

# Run DQN
start_time = time.time()
episode_durations, episode_returns, losses = run_episodes(train, qnet, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate, print_every=100, noise=state_noise)
duration = time.time() - start_time
print('run time in seconds: ', duration)

plt.plot(episode_returns)
plt.savefig('testplots_returns_curve.png')
plt.clf()
plt.plot(losses)
plt.savefig('testplots_loss_curve.png')
plt.clf()

# s=env.reset()
# policy.epsilon=0.
# plt.clf()
# done=False
# while True:
#     env.render()
#     if done:
#         break
#     action_idx, action = policy.sample_action(s,env._availableActionsInCurrentState())
#     s_next, r, done, info = env.step(action_idx)
#     s=s_next

# import matplotlib.pyplot as plt
# plt.clf()
# print('\n')

print('evaluation of learned policy on trainset')
policy.epsilon=0.
EvaluatePolicy(env, policy, env.world_pool, print_runs=False, save_plots=False)

# print('evaluation of learned policy on full set')
# policy.epsilon=0.
# EvaluatePolicy(env, policy, env.all_worlds, print_runs=False, save_plots=False)

# print('evaluation of random policy on full set')
# policy.epsilon=1.
# EvaluatePolicy(env, policy, env.all_worlds, print_runs=False, save_plots=False)

# print('evaluation of left-up policy on trainset')
# policy=MinIndegreePolicy(env)
# EvaluatePolicy(env, policy, env.world_pool, print_runs=False, save_plots=False)