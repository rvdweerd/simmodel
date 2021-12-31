import argparse
from environments import GraphWorld
import numpy as np
import simdata_utils as su
import matplotlib.pyplot as plt
import torch
from dqn_utils import seed_everything, FastReplayMemory, train, run_episodes
from rl_models import QNetwork
from rl_utils import EvaluatePolicy, CreateDuplicatesTrainsets, GetFullCoverageSample
from rl_policy import EpsilonGreedyPolicyDQN
from rl_custom_worlds import GetCustomWorld
from Phase1_hyperparameters import GetHyperParams_DQN
import time
import os

ALL_WORLDS=[
    'Manhattan3x3_PauseFreezeWorld',
    'Manhattan3x3_PauseDynamicWorld',
    'Manhattan5x5_DuplicateSetA',
    'Manhattan5x5_DuplicateSetB',
    'Manhattan5x5_FixedEscapeInit',
    'Manhattan5x5_VariableEscapeInit',
    'MetroU3_e17tborder_FixedEscapeInit',
    'MetroU3_e17t31_FixedEscapeInit', 
    'MetroU3_e17t0_FixedEscapeInit', 
    'MetroU3_e17tborder_VariableEscapeInit'
]
ALL_STATE_REPR=[
    'et',
    'etUt',
    'ete0U0',
    'etUte0U0'
]
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device = ',device)

def Run_DQN_Experiment(args):
    world_name_in = args.world_name
    state_repr_in = args.state_repr
    num_seeds     = args.num_seeds
    TRAIN         = args.train
    EVALUATE      = args.eval
    if world_name_in == 'all':
        world_names = ALL_WORLDS
    else:
        assert world_name_in in ALL_WORLDS
        world_names=[ world_name_in ]
    if state_repr_in == 'all':
        state_reprs = ALL_STATE_REPR
    else:
        assert state_repr_in in ALL_STATE_REPR
        state_reprs =[ state_repr_in ]
    for state_repr in state_reprs:
        for world_name in world_names:    
            # Setup
            #env=GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc='tensors')
            exp_rootdir='./results/DQN_testing/'+world_name+'/'+state_repr+'/'
            # Load hyperparameters
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

            # Train and evaluate
            env=GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc='tensors')
            dim_in = env.state_encoding_dim
            dim_out = env.max_outdegree
            if TRAIN:
                best_result=-1e6
                len_seeds=[]
                ret_seeds=[]
                er_seeds =[]
                for seed in range(num_seeds):
                    # Initialize
                    seed_everything(seed+42)
                    tensorboard_dir=exp_rootdir+'tensorboard/DQN'+str(seed+1)
                    memory = FastReplayMemory(mem_size,dim_in)
                    qnet = QNetwork(dim_in, dim_out, dims_hidden_layers).to(device)
                    policy = EpsilonGreedyPolicyDQN(qnet, env, eps_0=eps_0, eps_min=eps_min, eps_cutoff=cutoff)

                    # Run DQN
                    start_time = time.time()
                    episode_durations, episode_returns, losses, best_model = run_episodes(train, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate, print_every=100, noise=state_noise, logdir=tensorboard_dir)
                    duration = time.time() - start_time
                    print('run time in seconds: ', duration)

                    policy.model=best_model.to(device)
                    policy.epsilon=0.
                    lengths, returns, captures = EvaluatePolicy(env, policy, env.world_pool, print_runs=False, save_plots=False, logdir=exp_rootdir)    
                    # Results admin
                    len_seeds.append(np.mean(lengths))
                    ret_seeds.append(np.mean(returns))
                    er_seeds.append(1-np.sum(captures)/len(captures))
                    print('New result:',ret_seeds[-1],'Best result so far:',best_result)
                    if ret_seeds[-1] > best_result:
                        best_result = ret_seeds[-1]
                        torch.save(best_model.state_dict(), exp_rootdir+'Model_best')
                        print('saving best model...')
                OutputFile= exp_rootdir+'Results_over_seeds.txt'
                OF = open(OutputFile, 'w')
                def printing(text):
                    print(text)
                    OF.write(text + "\n")
                np.set_printoptions(formatter={'float':"{0:0.1f}".format})
                printing('Results over seeds')
                printing('Deterministic policy: '+str(policy.deterministic))
                printing('------------------------------------------------')
                printing('Avg er: '+str(np.mean(er_seeds))+' +/- '+str(np.std(er_seeds)))
                printing('  er_seeds: ' + str(er_seeds))
                printing('Avg epi lengths: '+str(np.mean(len_seeds))+' +/- '+str(np.std(len_seeds)))
                printing('  len_seeds: '+ str(len_seeds))
                printing('Avg returns: '+str(np.mean(ret_seeds))+' +/- '+str(np.std(ret_seeds)))
                printing('  ret_seeds: '+ str(ret_seeds))
                OF.close()

            if EVALUATE:
                qnet_best = QNetwork(dim_in, dim_out, dims_hidden_layers).to(device)
                qnet_best.load_state_dict(torch.load(exp_rootdir+'Model_best'))
                policy = EpsilonGreedyPolicyDQN(qnet_best, env, eps_0=eps_0, eps_min=eps_min, eps_cutoff=cutoff)
                policy.epsilon=0.
                lengths, returns, captures = EvaluatePolicy(env, policy, env.world_pool, print_runs=False, save_plots=False, logdir=exp_rootdir)
                plotlist = GetFullCoverageSample(returns, env.world_pool, bins=10, n=10)
                EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=exp_rootdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    

    # Model hyperparameters
    parser.add_argument('--world_name', default='Manhattan3x3_PauseFreezeWorld', type=str, 
                        help='Environment to run',
                        choices=[
                            'all',
                            'Manhattan3x3_PauseFreezeWorld',
                            'Manhattan3x3_PauseDynamicWorld',
                            'Manhattan5x5_FixedEscapeInit',
                            'Manhattan5x5_VariableEscapeInit',
                            'Manhattan5x5_DuplicateSetA',
                            'Manhattan5x5_DuplicateSetB',
                            'MetroU3_e17tborder_FixedEscapeInit',
                            'MetroU3_e17tborder_VariableEscapeInit',
                            'MetroU3_e17t31_FixedEscapeInit',
                            'MetroU3_e17t0_FixedEscapeInit' ])
    parser.add_argument('--state_repr', default='et', type=str, 
                        help='Which part of the state is observable',
                        choices=[
                            'et',
                            'etUt',
                            'ete0U0',
                            'etUte0U0',
                            'all' ])
    parser.add_argument('--num_seeds', default=1, type=int)
    parser.add_argument('--train', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('--eval', type=lambda s: s.lower() in ['true', 't', 'yes', '1'])

    args=parser.parse_args()
    Run_DQN_Experiment(args)
