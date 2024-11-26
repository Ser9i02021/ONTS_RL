import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from collections import namedtuple, deque
import pickle
import torch.nn as nn
import time


# Define the DQN Model
class DQN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class ONTSEnv:
    def __init__(self, u__job_priorities, q__energy_consumption_per_job, y_min_per_job, 
                 y_max_per_job, t_min_per_job, t_max_per_job, p_min_per_job, p_max_per_job,
                 w_min_per_job, w_max_per_job, r__energy_available_at_time_t, gamma, Vb, Q,
                 p, e, max_steps=None):
        
        self.u__job_priorities = u__job_priorities
        self.q__energy_consumption_per_job = q__energy_consumption_per_job
        self.y_min_per_job = y_min_per_job
        self.y_max_per_job = y_max_per_job
        self.t_min_per_job = t_min_per_job
        self.t_max_per_job = t_max_per_job
        self.p_min_per_job = p_min_per_job
        self.p_max_per_job = p_max_per_job
        self.w_min_per_job = w_min_per_job
        self.w_max_per_job = w_max_per_job
        self.r__energy_available_at_time_t = r__energy_available_at_time_t
        self.gamma = gamma 
        self.Vb = Vb
        self.Q = Q
        self.p = p
        self.e = e
        self.SoC_t = self.p

        self.J, self.T = len(u__job_priorities), len(r__energy_available_at_time_t)
        self.max_steps = max_steps if max_steps is not None else self.T
        self.x__state = None
        self.phi__state = None
        self.steps_taken = 0
        self.reset()
    
    def reset(self):
        self.x__state = np.zeros((self.J, self.T), dtype=int)
        self.phi__state = np.zeros((self.J, self.T), dtype=int)
        self.steps_taken = 0
        self.SoC_t = self.p
        return self.x__state.flatten()
    
    def step(self, action):
        job, time_step = divmod(action, self.T)
        self.steps_taken += 1
        self.x__state[job, time_step] = 1 - self.x__state[job, time_step]
        self.build_phi_matrix()  # Auxiliary matrix to check constraints
        reward, energy_exceeded = self.calculate_reward()
        done = energy_exceeded or self.steps_taken >= self.max_steps
        return self.x__state.flatten(), reward, done
    
    def build_phi_matrix(self):
        for j in range(self.J):
            for t in range(self.T):
                if t == 0:
                    if self.x__state[j, t] > self.phi__state[j, t]:
                        self.phi__state[j, t] = 1
                else:
                    if (self.x__state[j, t] - self.x__state[j, t-1]) > self.phi__state[j, t]:
                        self.phi__state[j, t] = 1
                    if self.phi__state[j, t] > (2 - self.x__state[j, t] - self.x__state[j, t-1]):
                        self.phi__state[j, t] = 0
                
                if self.phi__state[j, t] > self.x__state[j, t]:
                    self.phi__state[j, t] = 0

    def check_energy_constraints(self):
        for t in range(self.T):
            totalEnergyRequiredAtTimeStep_t = 0
            for j in range(self.J):
                totalEnergyRequiredAtTimeStep_t += self.x__state[j][t] * self.q__energy_consumption_per_job[j]
            
            if totalEnergyRequiredAtTimeStep_t > self.r__energy_available_at_time_t[t] + (self.gamma * self.Vb):
                return -1, False  # Penalize for exceeding max energy available at that time step
            
            exceedingPower = self.r__energy_available_at_time_t[t] - totalEnergyRequiredAtTimeStep_t
            i_t = exceedingPower / self.Vb
            self.SoC_t = self.SoC_t + (i_t * self.e) / (60 * self.Q)

            if self.SoC_t > 1:
                return -1, False  # Penalize for exceeding max state of charge at that time step
            
        return 0, False

    def check_job_constraints(self):
        acc_reward = 0
        for j in range(self.J):
            # w_min, w_max ((w_min, w_max] is the interval a job can be active)
            for tw in range(self.w_min_per_job[j]):
                if self.x__state[j, tw] == 1:
                    acc_reward -= 1  # Penalize for activating a job at a disallowed time step (below min)
                
            for tw in range(self.w_max_per_job[j], self.T):
                if self.x__state[j, tw] == 1:
                    acc_reward -= 1  # Penalize for activating a job at a disallowed time step (above max)
                
            # y_min, y_max (Min and Max times a job can be active)
            sum_l = 0
            for t in range(self.T):
                sum_l += self.phi__state[j, t]
            if sum_l < self.y_min_per_job[j]:
                acc_reward -= 1   # Penalize for a job not having been executed at least "y_min_per_job" times
            if sum_l > self.y_max_per_job[j]:
                acc_reward -= 1   # Penalize for a job having been executed more than "y_max_per_job" times
            
            # t_min, t_max (continuous job execution) 
            for t in range(self.T - self.t_min_per_job[j] + 1):
                tt_sum = 0
                for tt in range(t, t + self.t_min_per_job[j]):
                    tt_sum += self.x__state[j, tt] 
                if tt_sum < self.t_min_per_job[j] * self.phi__state[j, t]:
                    acc_reward -= 1   # Penalize for a job not running continuously for its minimum period
            
            for t in range(self.T - self.t_max_per_job[j]):
                tt_sum = 0
                for tt in range(t, t + self.t_max_per_job[j] + 1):
                    tt_sum += self.x__state[j, tt]
                if tt_sum > self.t_max_per_job[j]:
                    acc_reward -= 1   # Penalize for a job running continuously for more than its maximum period

            # p_min, p_max (periodic job execution)            
            for t in range(self.T - self.p_min_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_min_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l > 1:
                    acc_reward -= 1   # Penalize for a job not having been executed periodically for at least every "p_min_per_job" time steps
                
            for t in range(self.T - self.p_max_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_max_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l < 1:
                    acc_reward -= 1   # Penalize for a job having been executed periodically for more than every "p_max_per_job" time steps
        
        return acc_reward
    
    def calculate_reward(self):
        rewardSum = 0
        reward, done = self.check_energy_constraints()
        rewardSum += reward        
        if not done:
            reward = self.check_job_constraints()
            rewardSum += reward
            for j in range(self.J):
                for t in range(self.T):
                    # Reward is only positive if all restrictions are met
                    if rewardSum == 0:
                        rewardSum += 10 * (self.u__job_priorities[j] * self.x__state[j, t]) * (self.r__energy_available_at_time_t[t] - self.q__energy_consumption_per_job[j])

            return rewardSum, False
        
        return rewardSum, done

# Double DQN Training Function
def train_dqn(env, policy_net=None, target_net=None, episodes=500, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128, target_update=10):
    n_actions = env.J * env.T
    n_inputs = env.J * env.T
    
    # Initialize policy and target networks
    policy_net = DQN(n_inputs, n_actions) if policy_net is None else policy_net
    target_net = DQN(n_inputs, n_actions) if target_net is None else target_net
    target_net.load_state_dict(policy_net.state_dict())  # Copy initial weights to the target network
    
    optimizer = optim.Adam(policy_net.parameters())
    memory = deque(maxlen=10000)
    episode_durations = []
    epsilon = eps_start

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            epsilon = max(epsilon * eps_decay, eps_end)
            action = select_action_dqn(env, policy_net, state, epsilon)
            next_state, reward, done = env.step(action)
            total_reward += reward
            memory.append(Experience(torch.tensor([state], dtype=torch.float), torch.tensor([[action]], dtype=torch.long), torch.tensor([next_state], dtype=torch.float), torch.tensor([reward], dtype=torch.float)))
            optimize_model_dqn(policy_net, target_net, optimizer, memory, gamma, batch_size)  # Use both policy and target networks for optimization
            state = next_state
            if done:
                episode_durations.append(total_reward)
                break
        
        # Update target network every `target_update` episodes
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())  # Update target network to match policy network

    return policy_net, target_net


# Optimize model for Double DQN
def optimize_model_dqn(policy_net, target_net, optimizer, memory, gamma, batch_size):
    if len(memory) < batch_size:
        return
    experiences = random.sample(memory, batch_size)
    batch = Experience(*zip(*experiences))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    
    # Compute the Q-values of the current state using the policy network
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute the Q-values of the next state using the target network
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    
    # Calculate the expected Q-values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Use Huber loss (smooth_l1_loss) to handle outliers
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Select action for Double DQN
def select_action_dqn(env, policy_net, state, epsilon):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            return policy_net(torch.tensor([state], dtype=torch.float)).max(1)[1].view(1, 1).item()
    else:
        return random.randrange(env.J * env.T)

# Select action for Double DQN
def select_action_dqn(env, policy_net, state, epsilon):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            return policy_net(torch.tensor([state], dtype=torch.float)).max(1)[1].view(1, 1).item()
    else:
        return random.randrange(env.J * env.T)

def evaluate_dqn_model(env, policy_net, episodes=1):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()
        while True:
            action = select_action_dqn(env, policy_net, state, epsilon=0)
            state, reward, done = env.step(action)
            if done:
                break
        total_rewards += reward
        #print(f"Episode {episode+1}: Reward: {reward}")
        #print(state)
        #print()
    average_reward = total_rewards / episodes
    print(f"Average Reward over {episodes} episodes: {average_reward}")
    return average_reward


# Experience Tuple
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Define the ONTSEnv instance
u__job_priorities = np.array([3, 2, 1])
q__energy_consumption_per_job = np.array([1, 2, 1])
y_min_per_job = [1, 1, 1] 
y_max_per_job = [3, 4, 5]
t_min_per_job = [1, 1, 1]
t_max_per_job = [3, 4, 3]
p_min_per_job = [1, 1, 1]
p_max_per_job = [4, 5, 5]
w_min_per_job = [1, 1, 1]
w_max_per_job = [4, 5, 4]
r__energy_available_at_time_t = np.array([3, 3, 3, 3, 3])
gamma = 0.5
Vb = 1
Q = 10
p = 0.1
e = 0.9


'''# Store DQN policy and target networks into binary files
with open('policy_dqn.pkl', 'wb') as file:
    pickle.dump(policy_net, file)
with open('target_dqn.pkl', 'wb') as file:
    pickle.dump(target_net, file)
'''

sum = 0
for _ in range(10):
    env = ONTSEnv(u__job_priorities, q__energy_consumption_per_job, y_min_per_job, y_max_per_job, t_min_per_job, t_max_per_job, p_min_per_job, p_max_per_job, w_min_per_job, w_max_per_job, r__energy_available_at_time_t, gamma, Vb, Q, p, e)
    env.reset()
    # Train Double DQN agent
    policy_net, target_net = train_dqn(env, episodes=2000)
    # Evaluate the trained Double DQN model
    avg_rew = evaluate_dqn_model(env, policy_net, episodes=10)
    sum += avg_rew

print()
print(sum / 10)

