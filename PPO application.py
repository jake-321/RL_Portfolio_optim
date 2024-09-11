import yfindata as yf
from portfolio_env_PPO import PortfolioEnv
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch as T
import torch.distributions as dist
from ppo_torch_fin import PPOAgent

input_data = yf.ohlc_data['Adj Close'].values
env = PortfolioEnv(input_data)
initial_input = env.reset()
initial_input = T.transpose(T.tensor(initial_input),0,1).reshape(1,70)
agent = PPOAgent(n_actions = 7, input_dims = initial_input.shape, gamma = 0.99, alpha = 0.0003, 
                 gae_lambda = 0.95, policy_clip = 0.2, batch_size = 7, 
                 N = 28, n_epochs = 10)

N = 28
batch_size = 7
n_epochs = 4
alpha = 0.0003
n_games = 3000
    
best_score = env.reward_range[0]
score_history = []
learn_iters = 0
avg_score = 0
n_steps = 0
avglst = []

for i in range(n_games):
    observation = T.tensor(initial_input).reshape(1,70)
    done = False
    score = 0
    while not done:
        org_action, action, prob, val = agent.choose_action(observation)
        val = val.tolist()
        observation_, reward, done, info = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(observation, org_action, action, prob, val, reward, done)
        
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1

        observation = T.transpose(T.tensor(observation_),0,1).reshape(1,70)
        
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    avglst.append(avg_score)
    env.current_step = 10
    if avg_score > best_score:
        best_score = avg_score
        
    print('episode', i+1, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)

import matplotlib.pyplot as plt

plt.plot(np.arange(408),avglst)
plt.grid()
plt.title('Running avg of previous 100scores')
plt.axvline(color = 'black')
#plt.axhline(color = 'black')
plt.xlabel('epi')
plt.ylabel('avg score')
plt.show()