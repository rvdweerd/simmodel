import gym
from stable_baselines3.common.env_checker import check_env
from environments import GraphWorld
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
world_name='MetroU3_e17_FixedEscapeInit'
env=GetCustomWorld(world_name, make_reflexive=True, state_repr='etUt', state_enc='tensor')

#check_env(env)
# s=env.reset()
# env.render()
# print(s)

class NpWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        obs = np.array(observation).astype(np.float64)
        return obs
env = NpWrapper(env)

#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DQN

#model = DQN("MlpPolicy", env, verbose=0, tensorboard_log="./sb3_dqn_tensorboard/")
#model.learn(total_timesteps=100000, log_interval=1000)

#model = PPO.load('./models/sb3/ppo_'+world_name)

policy_kwargs = dict(activation_fn = torch.nn.Tanh, net_arch = [dict(pi = [64, 64], vf = [64, 64])])
policy_kwargs = dict(activation_fn = torch.nn.Tanh,net_arch = [dict(pi = [128,128,64], vf = [128,128,64])])
model = PPO('MlpPolicy', env, policy_kwargs = policy_kwargs, verbose=0, tensorboard_log="./sb3_ppo_tensorboard/")
print_parameters(model.policy)
#model.learn(total_timesteps=150000)
#model.save('./models/sb3/ppo_'+world_name)

from stable_baselines3.common.evaluation import evaluate_policy
N_eval=10000
#rewards, epi_lengths = evaluate_policy(model, env, n_eval_episodes=N_eval, deterministic=False, return_episode_rewards=True)
#print(f"mean_reward={np.mean(rewards):.2f} +/- {np.std(rewards)}")
#print(f"mean_lengths={np.mean(epi_lengths):.2f} +/- {np.std(epi_lengths)}")

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
        env.render(fname='./images/sb3/example_'+str(i)+'_t=')
    dones=False
    R=0
    steps=0
    GAM=1.
    while not dones:
        action, _states = model.predict(obs)
        old_obs=obs
        old_obs_nodes=env.state
        obs, rewards, dones, info = env.step(action.item())
        obs_nodes=env.state
        R+=rewards*GAM
        GAM*=gam
        steps+=1
        if i%save_every == 0:
            with torch.no_grad():
                old_obs = torch.tensor(old_obs)[None,:].to(device)
                all_actions = [i for i in range(env.max_outdegree)]
                all_actions = torch.tensor(all_actions).to(device)
                action_prob1 = torch.exp(model.policy.evaluate_actions(old_obs,all_actions)[1])
                action_prob2 = torch.exp(model.policy.get_distribution(old_obs).log_prob(all_actions))
            print('s:',old_obs_nodes,'a:',action,'action_probs',action_prob2.detach().cpu().numpy(), 'r',rewards, 's_',obs_nodes)
            env.render(fname='./images/sb3/example_'+str(i)+'_t=')
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

