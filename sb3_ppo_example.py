import gym
from stable_baselines3.common.env_checker import check_env
from environments import GraphWorld
import simdata_utils as su
from rl_custom_worlds import GetCustomWorld
import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Select graph world
#env=GetCustomWorld('Manhattan3x3_PauseFreezeWorld', make_reflexive=True, state_repr='et', state_enc='nodes')
env=GetCustomWorld('Manhattan3x3_PauseDynamicWorld', make_reflexive=False, state_repr='et', state_enc='nodes')
# check_env(env)
# s=env.reset()
# env.render()
# print(s)

class NpWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        obs = np.array(observation).astype(np.float)
        return obs
env = NpWrapper(env)

#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

#model = PPO.load('./models/sb3/ppo_PauseFreezeWorld_reflexive')
model = PPO.load('./models/sb3/ppo_PauseDynamicWorld_reflexive')
#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./sb3_ppo_tensorboard/")
#model.learn(total_timesteps=50000)
#model.save('./models/sb3/ppo_PauseDynamicWorld_reflexive')

from stable_baselines3.common.evaluation import evaluate_policy
N_eval=20000
# rewards, epi_lengths = evaluate_policy(model, env, n_eval_episodes=N_eval, deterministic=False, return_episode_rewards=True)
# print(f"mean_reward={np.mean(rewards):.2f} +/- {np.std(rewards)}")
# print(f"mean_lengths={np.mean(epi_lengths):.2f} +/- {np.std(epi_lengths)}")

np.set_printoptions(formatter={'float':"{0:0.2f}".format})
escape_count=0
save_every=10000
gam=1.
G=[]
L=[]
for i in range(N_eval):
    obs = env.reset()
    if i%save_every == 0:
        print('----------\nRun',i)
        env.render(fname='./images/sb3/example_'+str(i)+'_t=')
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
            action_prob1 = torch.exp(model.policy.evaluate_actions(torch.tensor(old_obs).to(device).long(),torch.tensor([0,1,2,3,4]).to(device))[1])
            action_prob2 = torch.exp(model.policy.get_distribution(torch.tensor(old_obs).to(device).long()).log_prob(torch.tensor([0,1,2,3,4]).to(device)))
            print('s:',old_obs,'a:',action,'action_probs',action_prob2.detach().cpu().numpy(), 'r',rewards, 's_',obs)
            env.render(fname='./images/sb3/example_'+str(i)+'_t=')
    G.append(R)
    L.append(steps)
    if rewards>0:
        escape_count+=1
    if i%save_every == 0:
        print('Running success ratio: {:.2f}'.format(escape_count/(i+1)))
print('Overall results:')
print('Escape ratio:',escape_count/N_eval)
print('Return mean:',np.mean(G),'std:',np.std(G))
print('Episode lengths mean:',np.mean(L),'std:',np.std(L))

