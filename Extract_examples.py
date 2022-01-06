# This file compares performances of selected (max 3) policies for a selected environment
# and saves random illustrative rollouts in results/_Examples
#
# NOTE: (need to fix) due to structure of PPO-RNN code, need to adjust parsed arguments
# either in launch.json or default values in Phase1_experiments_PPO_RNN.py

import random
from dataclasses import dataclass
from modules.rl.rl_policy import RandomPolicy, ShortestPathPolicy, EpsilonGreedyPolicyDQN, EpsilonGreedyPolicySB3_PPO, EpsilonGreedyPolicyLSTM_PPO2
from modules.rl.rl_utils import EvaluatePolicy, GetFullCoverageSample, NpWrapper
from modules.rl.rl_custom_worlds import CreateWorlds
from modules.rl.rl_models import QNetwork
from Phase1_hyperparameters import GetHyperParams_DQN, GetHyperParams_SB3PPO, GetHyperParams_PPO_RNN
from Phase1_experiments_PPO_RNN import start_or_resume_from_checkpoint
import numpy as np
import torch
from stable_baselines3 import PPO
device = 'cuda' if torch.cuda.is_available() else 'cpu'

run_world_names = [
    'Manhattan3x3_PauseFreezeWorld',            # 0
    'Manhattan3x3_PauseDynamicWorld',           # 1
    'Manhattan5x5_DuplicateSetA',               # 2
    'Manhattan5x5_DuplicateSetB',               # 3
    'Manhattan5x5_FixedEscapeInit',             # 4
    'Manhattan5x5_VariableEscapeInit',          # 5
    'MetroU3_e17tborder_FixedEscapeInit',       # 6
    'MetroU3_e17t31_FixedEscapeInit',           # 7
    'MetroU3_e17t0_FixedEscapeInit',            # 8
    'MetroU3_e17tborder_VariableEscapeInit'     # 9
]

world_name='Manhattan5x5_VariableEscapeInit'
state_repr='etUt'
policy_names=['DQN','PPO','RNN-PPO']
num_p=len(policy_names)
assert (num_p > 0 and num_p < 4)

worlds_n = CreateWorlds(run_world_names, make_reflexive=True, state_repr=state_repr, state_enc='nodes')
worlds_n = dict(zip(run_world_names,worlds_n))
worlds_t = CreateWorlds(run_world_names, make_reflexive=True, state_repr=state_repr, state_enc='tensors')
worlds_t = dict(zip(run_world_names,worlds_t))

env_n=worlds_n[world_name]
env_t=worlds_t[world_name]

# Load DQN policy
exp_rootdir='./results/DQN/'+world_name+'/'+state_repr+'/'
hp=GetHyperParams_DQN(world_name)
dims_hidden_layers  = hp['dims_hidden_layers'][state_repr]
batch_size          = hp['batch_size'][state_repr]
mem_size            = hp['mem_size'][state_repr]
discount_factor     = .9
learn_rate          = hp['learning_rate'][state_repr]
num_episodes        = hp['num_episodes'][state_repr]
eps_0               = hp['eps_0'][state_repr]
eps_min             = hp['eps_min'][state_repr]
cutoff_factor       = hp['cutoff_factor'][state_repr]
cutoff              = cutoff_factor *  num_episodes # lower plateau reached and maintained from this point onward
state_noise         = False
dim_in = env_t.state_encoding_dim
dim_out = env_t.max_outdegree
qnet_best = QNetwork(dim_in, dim_out, dims_hidden_layers).to(device)
qnet_best.load_state_dict(torch.load(exp_rootdir+'Model_best'))

# Load PPO policy
exp_rootdir='./results/PPO/'+world_name+'/'+state_repr+'/'
hp=GetHyperParams_SB3PPO(world_name)
actor_dims        = hp['actor'][state_repr]
critic_dims       = hp['critic'][state_repr]
activation        = hp['activation'][state_repr]
num_seeds         = hp['num_seeds'][state_repr]
total_steps       = hp['total_steps'][state_repr]
eval_deterministic= hp['eval_determ'][state_repr]
sample_multiplier = hp['sampling_m'][state_repr]
env_ppo = NpWrapper(env_t)
ppo_model = PPO.load(exp_rootdir+'Model_best')

# Load RNN-PPO policy
#from Phase1_experiments_PPO_RNN import StopConditions
@dataclass
class StopConditions():
    """
    Store parameters and variables used to stop training. 
    """
    best_reward: float = -1e6
    fail_to_improve_count: int = 0
    max_iterations: int = 1000000
hp=GetHyperParams_PPO_RNN(world_name)
HIDDEN_SIZE=hp['HIDDEN_SIZE'][state_repr]
EVAL_DETERMINISTIC=hp['EVAL_DETERMINISTIC'][state_repr]
@dataclass
class HyperParameters():
    scale_reward:         float = 1.
    min_reward:           float = 1.
    hidden_size:          float = 1.
    batch_size:           int   = 1
    discount:             float = 1.
    gae_lambda:           float = 1.
    ppo_clip:             float = 1.
    ppo_epochs:           int   = 1
    max_grad_norm:        float = 1.
    entropy_factor:       float = 1.
    actor_learning_rate:  float = 1.
    critic_learning_rate: float = 1.
    recurrent_seq_len:    int = 1
    recurrent_layers:     int = 1 
    rollout_steps:        int = 1
    parallel_rollouts:    int = 1
    patience:             int = 1
    # Apply to continous action spaces only 
    trainable_std_dev:    bool = False    
    init_log_std_dev:     float = 1.
actor, critic, actor_optimizer, critic_optimizer, max_checkpoint_iteration, stop_conditions=start_or_resume_from_checkpoint()
@dataclass
class LSTP_PPO_MODEL():
    lstm_hidden_dim:    int 
    pi:                 torch.nn.modules.module.Module = None
    v:                  torch.nn.modules.module.Module = None
lstm_ppo_model = LSTP_PPO_MODEL(lstm_hidden_dim=HIDDEN_SIZE, pi=actor.to(device), v=critic.to(device))

policies={
    'random'  : RandomPolicy(env_t),
    'shortest': ShortestPathPolicy(env_t, weights = 'equal'),
    'mindeg'  : ShortestPathPolicy(env_t, weights = 'min_indegree'),
    'TabQL'   : 2,
    'DQN'     : EpsilonGreedyPolicyDQN(qnet_best, env_t, eps_0=0., eps_min=0., eps_cutoff=0),
    'PPO'     : EpsilonGreedyPolicySB3_PPO(env_ppo, ppo_model, deterministic=eval_deterministic),
    'RNN-PPO' : EpsilonGreedyPolicyLSTM_PPO2(env_t,lstm_ppo_model, deterministic=EVAL_DETERMINISTIC),
}
env=[]
policy=[]
logdirs=[]
lengths=[]
returns=[]
captures=[]
for i in range(num_p):
    env.append(env_t)
    policy.append(policies[policy_names[i]])
    if policy_names[i] in ['random','shortest','mindeg','tabQL']: 
        env[i]=env_n
    elif policy_names[i] in ['PPO']:
        env[i]=env_ppo
    logdirs.append('results/_Examples/'+world_name+'/'+state_repr+'/'+policy_names[i])
    len_, ret_, cap_ = EvaluatePolicy(env[i],policy[i], env[i].world_pool, print_runs=False, save_plots=False, logdir=logdirs[i])
    lengths.append(len_)
    returns.append(ret_)
    captures.append(cap_)

# Extract worlds of interest
returns=np.array(returns)
example_pools1=[[] for i in range(num_p)]
example_pools2=[[] for i in range(num_p)]
selected_pool=[]
for i in range(returns.shape[1]):
    maxind = np.where(returns[:,i] == np.max(returns[:,i]))[0]
    if len(maxind)==1:
        if np.min(returns[:,i])>0:
            example_pools1[maxind.item()].append(env[0].world_pool[i])
        else:
            example_pools2[maxind.item()].append(env[0].world_pool[i])
for i in range(num_p):
    if len(example_pools1[i])>0:
        selected_pool.append(random.choice(example_pools1[i]))
    if len(example_pools2[i])>0:
        selected_pool.append(random.choice(example_pools2[i]))
res_overview=[[selected_pool]]
for i in range(num_p):
    _,r,_ = EvaluatePolicy(env[i],policy[i], selected_pool, print_runs=False, save_plots=True, logdir=logdirs[i])
    res_overview.append(r)

ofile='results/_Examples/'+world_name+'/'+state_repr+'/Results_overview.txt'
OF = open(ofile, 'w')
def printing(text):
    print(text)
    OF.write(text + "\n")
np.set_printoptions(formatter={'float':"{0:0.1f}".format})
printing('WORLDS: '+str(res_overview[0]))
for i in range(num_p):
    printing(policy_names[i]+': '+str(res_overview[i+1]))
