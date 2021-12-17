import gym
from environments import GraphWorld
import simdata_utils as su
from rl_custom_worlds import GetCustomWorld
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Select graph world
world_name='Manhattan3x3_PauseFreezeWorld'
#world_name='Manhattan3x3_PauseDynamicWorld'
#world_name='Manhattan5x5_FixedEscapeInit'
#world_name='Manhattan5x5_VariableEscapeInit'
#world_name='MetroU3_e17_FixedEscapeInit'
env=GetCustomWorld(world_name, make_reflexive=False, state_repr='et', state_enc='nodes')

# check_env(env)
# s=env.reset()
# env.render()
# print(s)

class NpWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        obs = np.array(observation).astype(np.float)
        return obs
env = NpWrapper(env)

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpLstmPolicy, MlpLnLstmPolicy

import tensorflow as tf
cur_sess = tf.compat.v1.Session()

#pol=MlpLstmPolicy(sess=cur_sess, ob_space=env.observation_space, ac_space=env.action_space, n_env=1, n_steps=10, n_batch=10, n_lstm=8)
#model = PPO.load('./models/sb3/ppo_'+world_name)
#model = PPO2('MlpPolicy', env, verbose=0, tensorboard_log="./sb2_ppo_tensorboard/")
model = PPO2('MlpLnLstmPolicy', env, \
        verbose=0, \
        nminibatches=1, \
        tensorboard_log="./sb2_ppo_tensorboard/", \
        seed=42
    )
model.learn(total_timesteps=50000)

model.save('./models/sb2/ppo_'+world_name)

#from stable_baselines3.common.evaluation import evaluate_policy
N_eval=10000
# rewards, epi_lengths = evaluate_policy(model, env, n_eval_episodes=N_eval, deterministic=False, return_episode_rewards=True)
# print(f"mean_reward={np.mean(rewards):.2f} +/- {np.std(rewards)}")
# print(f"mean_lengths={np.mean(epi_lengths):.2f} +/- {np.std(epi_lengths)}")

np.set_printoptions(formatter={'float':"{0:0.2f}".format})
escape_count=0
save_every=2000
gam=1.
G=[]
L=[]
for i in range(N_eval):
    obs = env.reset()
    if i%save_every == 0:
        print('----------\nRun',i)
        env.render(fname='./images/sb2/example_'+str(i)+'_t=')
    dones=False
    R=0
    steps=0
    GAM=1.
    while not dones:
        action, _states = model.predict(obs)
        old_obs=obs
        obs, rewards, dones, info = env.step(action.item())
        R+=rewards*GAM
        GAM*=gam
        steps+=1
        if i%save_every == 0:
            #action_prob1 = torch.exp(model.policy.evaluate_actions(torch.tensor(old_obs).to(device).long(),torch.tensor([0,1,2,3,4]).to(device))[1])
            #action_prob2 = torch.exp(model.policy.get_distribution(torch.tensor(old_obs).to(device).long()).log_prob(torch.tensor([0,1,2,3,4]).to(device)))
            #print('s:',old_obs,'a:',action,'action_probs',action_prob2.detach().cpu().numpy(), 'r',rewards, 's_',obs)
            env.render(fname='./images/sb2/example_'+str(i)+'_t=')
    G.append(R)
    L.append(steps)
    if rewards>0:
        escape_count+=1
    if i%save_every == 0:
        print('Running success ratio: {:.2f}'.format(escape_count/(i+1)))
print('Overall results:')
print('Escape ratio:',escape_count/N_eval)
print('Episode lengths mean:',np.mean(L),'std:',np.std(L))
print('Return mean:',np.mean(G),'std:',np.std(G))

