from tqdm import tqdm as _tqdm
import numpy as np
#import random

def tqdm(*args, **kwargs):
    # Safety, do not overflow buffer
    return _tqdm(*args, **kwargs, mininterval=1)

def q_learning(env, policy, num_episodes, discount_factor=1.0, alpha_0 = 0.5, alpha_decay=0., print_episodes=False):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        policy: A behavior policy which allows us to sample actions with its sample_action method.
        Q: Q value function
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.

    Returns:
        A tuple (Q, stats).
        Q is a numpy array Q[s,a] -> state-action value.
        stats is a list of tuples giving the episode lengths and returns.
    """

    # Keeps track of useful statistics
    stats = []
    Q_tables = []

    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0

        if hasattr(env, 'custom_reset'):
            start_state = env.custom_reset()
        else:
            start_state = env.reset()
        done = False
        if print_episodes: print('episode '+str(i_episode)+' - [S'+str(start_state)+' ', end='')
        while not done:
            start_action = policy.sample_action(start_state)
            new_state, reward, done, _ = env.step(start_action)
            if print_episodes: 
                eps=policy.get_epsilon(start_state)
                print('eps('+'{:.2f}'.format(eps)+'),a('+str(start_action)+'),(r'+str(reward)+')]-> [S'+str(new_state)+' ', end='')
            
            if new_state not in policy.Q:
                policy.Q[new_state] = np.ones(policy.num_actions[new_state[0]]).astype(np.float32) * policy.initial_Q_values
            Qnew = np.max(policy.Q[new_state])
            if done:
                Qnew = 0
            if alpha_decay > 0:
                alpha = alpha_0 / policy.sa_count[start_state[0], start_action]**(alpha_decay)
            else:
                alpha = alpha_0

            policy.Q[start_state][start_action] = \
                policy.Q[start_state][start_action] + \
                alpha * (reward + discount_factor * Qnew - policy.Q[start_state][start_action])

            start_state = new_state
            i += 1
            R += (discount_factor**i) * reward
        if print_episodes: print('] - Steps:', i, 'Reward', R)
        stats.append((i, R))
        Q_tables.append(policy.Q.copy())

    episode_lengths, episode_returns = zip(*stats)
    metrics_vanilla = [episode_returns, episode_lengths]
    return policy.Q, metrics_vanilla, policy, Q_tables

def sarsa(env, policy, num_episodes, discount_factor=1.0, alpha_0 = 0.5, alpha_decay=0., print_episodes=False):
    """
    """

    # Keeps track of useful statistics
    stats = []
    Q_tables = []

    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0

        if hasattr(env, 'custom_reset'):
            start_state = env.custom_reset()
        else:
            start_state = env.reset()
        done = False
        if print_episodes: print('episode', i_episode, ' - ', end='')
        first_action = policy.sample_action(start_state)
        while not done:
            new_state, reward, done, _ = env.step(first_action)
            next_action = policy.sample_action(new_state)

            Qnew = policy.Q[new_state][next_action]
            if done:
                Qnew = 0
            if alpha_decay > 0:
                alpha = alpha_0 / policy.sa_count[start_state, first_action]**(alpha_decay)
            else:
                alpha = alpha_0

            policy.Q[start_state][first_action] = \
                policy.Q[start_state][first_action] + \
                alpha * (reward + discount_factor * Qnew - policy.Q[start_state][first_action])

            start_state = new_state
            first_action = next_action
            i += 1
            R += (discount_factor**i) * reward
        if print_episodes: print(' - steps:', i, 'Reward', R)
        stats.append((i, R))
        Q_tables.append(policy.Q.copy())

    episode_lengths, episode_returns = zip(*stats)
    metrics_vanilla = [episode_returns, episode_lengths]
    return policy.Q, metrics_vanilla, policy, Q_tables

def expected_sarsa(env, policy, num_episodes, discount_factor=1.0, alpha_0 = 0.5, alpha_decay=0., print_episodes=False):
    """
    """

    # Keeps track of useful statistics
    stats = []
    Q_tables = []

    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0

        if hasattr(env, 'custom_reset'):
            start_state = env.custom_reset()
        else:
            start_state = env.reset()
        done = False
        if print_episodes: print('episode', i_episode, ' - ', end='')
        while not done:
            start_action = policy.sample_action(start_state)
            new_state, reward, done, _ = env.step(start_action)

            Qnew = 0
            eps=policy.epsilon / policy.state_count[start_state]**policy.eps_decay
            amax = np.argmax(policy.Q[new_state])
            for a in range(env.nA[new_state]):
                Qnew += eps / env.nA[new_state] * policy.Q[new_state][a]
                if a == amax:
                    Qnew += (1-eps) * policy.Q[new_state][a]

            if done:
                Qnew = 0
            if alpha_decay > 0:
                alpha = alpha_0 / policy.sa_count[start_state, start_action]**(alpha_decay)
            else:
                alpha = alpha_0

            policy.Q[start_state][start_action] = \
                policy.Q[start_state][start_action] + \
                alpha * (reward + discount_factor * Qnew - policy.Q[start_state][start_action])

            start_state = new_state
            i += 1
            R += (discount_factor**i) * reward
        if print_episodes: print(' - steps:', i, 'Reward', R)
        stats.append((i, R))
        Q_tables.append(policy.Q.copy())

    episode_lengths, episode_returns = zip(*stats)
    metrics_vanilla = [episode_returns, episode_lengths]
    return policy.Q, metrics_vanilla, policy, Q_tables



def mc_q_learning(env, policy, num_episodes, discount_factor=1.0, alpha_0 = 0.5, alpha_decay=0., print_episodes=False):
    # Keeps track of useful statistics
    stats = []
    Q_tables = []
    Returns = []#[ [ [] for m in policy.Q.shape[1] ] for n in policy.Q.shape[0]]
    if isinstance(policy.Q,dict):
        for k,v in policy.Q.items():
            Returns.append([[] for _ in range(len(v))])
    else:
        Returns = [ [ [] for m in range(policy.Q.shape[1]) ] for n in range(policy.Q.shape[0])]
    for i_episode in tqdm(range(num_episodes)):
        i = 0
        R = 0
        SA_History = []
        R_History = []
        if hasattr(env, 'custom_reset'):
            start_state = env.custom_reset()
        else:
            start_state = env.reset()
        done = False
        if print_episodes: print('episode', i_episode, ' - ', end='')
        while not done:
            start_action = policy.sample_action(start_state)
            new_state, reward, done, _ = env.step(start_action)
            SA_History.append((start_state,start_action))
            R_History.append(R)
            start_state = new_state
            i += 1
            R += (discount_factor**i) * reward
        if print_episodes: print(' - steps:', i, 'Reward', R)
        G=0
        for t in reversed(range(len(SA_History))):
            G=discount_factor*G+R_History[t]
            if SA_History[t] not in SA_History[:t]:
                Returns[SA_History[t][0]][SA_History[t][1]].append(G)
                policy.Q[SA_History[t][0]][SA_History[t][1]] = np.mean(Returns[SA_History[t][0]][SA_History[t][1]])
        
        stats.append((i, R))
        Q_tables.append(policy.Q.copy())

    episode_lengths, episode_returns = zip(*stats)
    metrics_vanilla = [episode_returns, episode_lengths]
    return policy.Q, metrics_vanilla, policy, Q_tables

