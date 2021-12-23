import argparse
import gym
import simdata_utils as su
from stable_baselines3.common.env_checker import check_env
from environments import GraphWorld
from rl_policy import EpsilonGreedyPolicySB3_PPO
from Phase1_hyperparameters import GetHyperParams_SB3PPO
from rl_custom_worlds import GetCustomWorld
from rl_utils import EvaluatePolicy, print_parameters, GetFullCoverageSample, NpWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DQN
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def RunSB3_PPO_Experiment(args):
    world_name = args.world_name
    state_repr = args.state_repr
    TRAIN = args.TRAIN
    EVALUATE = args.EVAL
    
    # Setup
    env=GetCustomWorld(world_name, make_reflexive=True, state_repr=state_repr, state_enc='tensors')
    env = NpWrapper(env)
    hp=GetHyperParams_SB3PPO(world_name)
    actor_dims        = hp['actor'][env.state_representation]
    critic_dims       = hp['critic'][env.state_representation]
    activation        = hp['activation'][env.state_representation]
    num_seeds         = hp['num_seeds'][env.state_representation]
    total_steps       = hp['total_steps'][env.state_representation]
    eval_deterministic= hp['eval_determ'][env.state_representation]
    sample_multiplier = hp['sampling_m'][env.state_representation]
    exp_rootdir='./results/sb3-PPO/'+world_name+'/'+env.state_representation+'/'

    # Train and evaluate
    if TRAIN:
        policy_kwargs = dict(
                activation_fn = activation, 
                net_arch = [dict(pi = actor_dims, vf = critic_dims)]
            )
        best_result=-1e6
        len_seeds=[]
        ret_seeds=[]
        er_seeds =[]
        for i in range(num_seeds):
            model = PPO('MlpPolicy', env, \
                #learning_rate=1e-4,\
                seed=i,\
                #clip_range=0.1,\    
                #max_grad_norm=0.1,\
                policy_kwargs = policy_kwargs, verbose=1, tensorboard_log=exp_rootdir+"tensorboard/")
            print_parameters(model.policy)
            model.learn(total_timesteps = total_steps)
            #model.save(exp_rootdir+'Model_run'+str(i))

            policy=EpsilonGreedyPolicySB3_PPO(env, model, deterministic=eval_deterministic)
            lengths, returns, captures = EvaluatePolicy(env, policy, env.world_pool*sample_multiplier, print_runs=False, save_plots=False, logdir=exp_rootdir)    
            
            # Results admin
            len_seeds.append(np.mean(lengths))
            ret_seeds.append(np.mean(returns))
            er_seeds.append(1-np.sum(captures)/len(captures))
            print('New result:',ret_seeds[-1],'Best result so far:',best_result)
            if ret_seeds[-1] > best_result:
                best_result = ret_seeds[-1]
                model.save(exp_rootdir+'Model_best')
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
        model = PPO.load(exp_rootdir+'Model_best')
        policy=EpsilonGreedyPolicySB3_PPO(env, model, deterministic=eval_deterministic)
        lengths, returns, captures = EvaluatePolicy(env, policy, env.world_pool*sample_multiplier, print_runs=False, save_plots=False, logdir=exp_rootdir)
        plotlist = GetFullCoverageSample(returns, env.world_pool*sample_multiplier, bins=10, n=10)
        EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=exp_rootdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    

    # Model hyperparameters
    parser.add_argument('--world_name', default='Manhattan3x3_PauseFreezeWorld', type=str, 
                        help='Environment to run',
                        choices=[
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
                            'etUte0U0' ])
    parser.add_argument('--TRAIN', default=True, type=bool, 
                        help='Train a policy',
                        choices=[True, False])
    parser.add_argument('--EVAL', default=True, type=bool, 
                        help='Evaluate a (trained or saved) policy',
                        choices=[True, False])

    args=parser.parse_args()
    RunSB3_PPO_Experiment(args)

# from stable_baselines3.common.evaluation import evaluate_policy
# N_eval=10000
#rewards, epi_lengths = evaluate_policy(model, env, n_eval_episodes=N_eval, deterministic=False, return_episode_rewards=True)
#print(f"mean_reward={np.mean(rewards):.2f} +/- {np.std(rewards)}")
#print(f"mean_lengths={np.mean(epi_lengths):.2f} +/- {np.std(epi_lengths)}")