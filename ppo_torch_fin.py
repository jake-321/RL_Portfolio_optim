import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.org_actions = []
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype = np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.org_actions), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, org_action, action, probs, vals, reward, done):
            self.states.append(state)
            self.actions.append(action)
            self.probs.append(probs)
            self.vals.append(vals)
            self.rewards.append(reward)
            self.dones.append(done)
            self.org_actions.append(org_action)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.org_actions = []

class ActorNetwork(nn.Module):
    def __init__(self, alpha):
        super(ActorNetwork, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(70, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(16, 1)
        self.std_layer = nn.Linear(16,1)
        self.positive = nn.Softplus()
        
        self.actor = self.actor.double()
        self.mu_layer = self.mu_layer.double()
        self.std_layer = self.std_layer.double()
        self.positive = self.positive.double()

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)

    
    def forward(self, state):
        x = self.actor(state)
        mean = self.positive(self.mu_layer(x))
        std = self.positive(self.std_layer(x))

        distribution = T.cat((mean, std), dim=-1)
                       
        return distribution
        
class CriticNetwork(nn.Module):
    def __init__(self, alpha):
        super(CriticNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(70, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.critic = self.critic.double()

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        value = self.critic(state)
        return value.mean(dtype=T.float32)

class PPOAgent:
    def __init__(self, n_actions, input_dims, gamma, alpha, gae_lambda, policy_clip, batch_size, N, n_epochs):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(alpha)
        self.critic = CriticNetwork(alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, org_action, action, probs, vals, reward, done):
        self.memory.store_memory(state, org_action, action, probs, vals, reward, done)

    def choose_action(self, state):
        
        distribution = self.actor(state)
        distribution = distribution.squeeze(0).tolist()
        mean = distribution[0]
        std = distribution[1]
        m = nn.Softplus()
        action_dist = dist.Normal(mean, std) 
        org_action = m(action_dist.sample((7,)))
        for i in range(len(org_action)):
            if org_action[i] < 0:
                org_action[i] = 0
        
        org_action = org_action.detach().numpy().flatten()
        action = org_action / np.sum(org_action)
        prob = action_dist.log_prob(T.tensor(org_action)).tolist()
        value = self.critic(state)
        
        return org_action, action, prob, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, org_action_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype = np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to("cuda" if T.cuda.is_available() else "cpu")

            values = T.tensor(values).to("cuda" if T.cuda.is_available() else "cpu")
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype = T.double).to("cuda" if T.cuda.is_available() else "cpu")
                old_probs = T.tensor(old_prob_arr[batch]).to("cuda" if T.cuda.is_available() else "cpu")
                actions = T.tensor(action_arr[batch]).to("cuda" if T.cuda.is_available() else "cpu")
                org_actions = T.tensor(org_action_arr[batch]).to("cuda" if T.cuda.is_available() else "cpu")
                distribution = self.actor(T.transpose(states,0,1).reshape(7,70))

                dist_lst = []

                for i in range(len(distribution.tolist())):
                    normal_d = dist.Normal(distribution.tolist()[i][0], distribution.tolist()[i][1])
                    dist_lst.append(normal_d)
                
                critic_value = self.critic(T.transpose(states,0,1).reshape(7,70))
                
                critic_value = T.squeeze(critic_value)
                
                new_probs = []
                for j in range(len(dist_lst)):
                    new_pro = dist_lst[j].log_prob(actions)
                    new_probs.append(new_pro)
                
                nwp = []
                for l in range(len(new_probs)):
                    ak = new_probs[l].tolist()
                    nwp.append(ak)
                new_probs = T.tensor(nwp)
                prob_ratio = new_probs.exp()/old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimiser.zero_grad()
                self.critic.optimiser.zero_grad()
                total_loss.backward()
                self.actor.optimiser.step()
                self.critic.optimiser.step()
        
        self.memory.clear_memory()