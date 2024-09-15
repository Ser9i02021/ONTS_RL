import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import torch.optim as optim
import random
from collections import namedtuple, deque
from torch_geometric.data import Data, Batch
import pickle

# This imports all necessary libraries for the implementation.
# The key components include PyTorch, which is used for deep learning, and PyTorch Geometric (PyG) for graph neural networks (GNN).
# Optimizers, sampling, and experience replay memory are also imported to facilitate reinforcement learning (RL).

# Define the Graph Neural Network (GNN) model for policy learning.
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        # Three graph convolutional layers
        self.conv1 = GCNConv(in_channels, 128)  # First GCN layer
        self.conv2 = GCNConv(128, 128)          # Second GCN layer
        self.conv3 = GCNConv(128, 64)           # Third GCN layer
        # Fully connected layers to process the GCN output
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, out_channels)

    # Forward pass through the network
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Apply the first graph convolution, followed by ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Apply the second GCN layer with ReLU
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Apply the third GCN layer with ReLU
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        # Global mean pooling over the graph to collapse node information
        x = global_mean_pool(x, batch)
        # Pass through fully connected layers to output the action space
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# ONTSEnv class simulates the ONTS problem (task scheduling) environment.
class ONTSEnv:
    def __init__(self, u__job_priorities, q__energy_consumption_per_job, y_min_per_job, y_max_per_job, 
                 t_min_per_job, t_max_per_job, p_min_per_job, p_max_per_job, 
                 w_min_per_job, w_max_per_job, r__energy_available_at_time_t, gamma, Vb, Q, p, e, max_steps=None):
        
        # Initialize environment variables and constraints
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
        self.SoC_t = self.p  # Initialize state of charge

        self.J, self.T = len(u__job_priorities), len(r__energy_available_at_time_t)
        self.max_steps = max_steps if max_steps is not None else self.T  # Set maximum steps
        self.x__state = None
        self.phi__state = None
        self.steps_taken = 0
        self.reset()  # Initialize the environment

    # Reset the environment to an initial state
    def reset(self):
        # Reset state matrices for jobs and auxiliary variables
        self.x__state = np.zeros((self.J, self.T), dtype=int)
        self.phi__state = np.zeros((self.J, self.T), dtype=int)
        self.steps_taken = 0
        self.SoC_t = self.p  # Reset state of charge
        return self.x__state.flatten()

    # Step function takes an action, updates the environment, and returns reward and next state
    def step(self, action):
        job, time_step = divmod(action, self.T)  # Convert action to job and time_step
        self.steps_taken += 1
        self.x__state[job, time_step] = 1 - self.x__state[job, time_step]  # Flip the state of the job at time step
        self.build_phi_matrix()  # Update auxiliary matrix for job scheduling constraints
        reward, energy_exceeded = self.calculate_reward()  # Calculate the reward
        done = energy_exceeded or self.steps_taken >= self.max_steps  # Check if the episode is done
        return self.x__state, reward, done

    # Build auxiliary matrix to track job activation and deactivation constraints
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

    # Check energy constraints for job scheduling
    def check_energy_constraints(self):
        for t in range(self.T):
            totalEnergyRequiredAtTimeStep_t = 0
            for j in range(self.J):
                totalEnergyRequiredAtTimeStep_t += self.x__state[j][t] * self.q__energy_consumption_per_job[j]
            # Check if total energy exceeds the available energy
            if totalEnergyRequiredAtTimeStep_t > self.r__energy_available_at_time_t[t] + (self.gamma * self.Vb):
                return -1, False  # Penalize for exceeding energy
            # Calculate state of charge (SoC)
            exceedingPower = self.r__energy_available_at_time_t[t] - totalEnergyRequiredAtTimeStep_t
            i_t = exceedingPower / self.Vb
            self.SoC_t = self.SoC_t + (i_t * self.e) / (60 * self.Q)
            if self.SoC_t > 1:
                return -1, False  # Penalize for exceeding max SoC
        return 0, False  # No constraint violations

    # Check job-specific constraints such as minimum and maximum times, periods, etc.
    def check_job_constraints(self):
        acc_reward = 0
        for j in range(self.J):
            # Check if jobs are activated at the correct time intervals (w_min, w_max)
            for tw in range(self.w_min_per_job[j]):
                if self.x__state[j, tw] == 1:
                    acc_reward -= 1  # Penalize for activating at disallowed time
            for tw in range(self.w_max_per_job[j], self.T):
                if self.x__state[j, tw] == 1:
                    acc_reward -= 1  # Penalize for activating at disallowed time
                
            # Check minimum and maximum times job can be active (y_min, y_max)
            sum_l = 0
            for t in range(self.T):
                sum_l += self.phi__state[j, t]
            if sum_l < self.y_min_per_job[j]:
                acc_reward -= 1  # Penalize for not meeting minimum execution times
            if sum_l > self.y_max_per_job[j]:
                acc_reward -= 1  # Penalize for exceeding maximum execution times
            
            # Check continuous execution constraints (t_min, t_max)
            for t in range(self.T - self.t_min_per_job[j] + 1):
                tt_sum = 0
                for tt in range(t, t + self.t_min_per_job[j]):
                    tt_sum += self.x__state[j, tt]
                if tt_sum < self.t_min_per_job[j] * self.phi__state[j, t]:
                    acc_reward -= 1  # Penalize for not meeting continuous execution

            for t in range(self.T - self.t_max_per_job[j]):
                tt_sum = 0
                for tt in range(t, t + self.t_max_per_job[j] + 1):
                    tt_sum += self.x__state[j, tt]
                if tt_sum > self.t_max_per_job[j]:
                    acc_reward -= 1  # Penalize for exceeding maximum continuous execution
                
            # Check periodic execution constraints (p_min, p_max)
            for t in range(self.T - self.p_min_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_min_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l > 1:
                    acc_reward -= 1  # Penalize for not meeting periodic execution

            for t in range(self.T - self.p_max_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_max_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l < 1:
                    acc_reward -= 1  # Penalize for exceeding periodic execution
        return acc_reward

    # Calculate the reward based on job constraints and energy constraints
    def calculate_reward(self):
        rewardSum = 0
        reward, done = self.check_energy_constraints()  # Check energy constraints first
        rewardSum += reward        
        if not done:
            reward = self.check_job_constraints()  # Check job constraints next
            rewardSum += reward
            for j in range(self.J):
                for t in range(self.T):
                    # Reward is given for meeting all restrictions and based on job priorities
                    if rewardSum == 0:
                        rewardSum += 10 * (self.u__job_priorities[j] * self.x__state[j, t]) * (self.r__energy_available_at_time_t[t] - self.q__energy_consumption_per_job[j])
            return rewardSum, False  # No termination, continue
        return rewardSum, done  # Return reward and done status if exceeded energy

    # Get the graph representation of the environment (used by GNN)
    def get_graph(self):
        edge_index = self.create_edges()  # Create edge connections for time steps
        x = torch.tensor(self.x__state.flatten(), dtype=torch.float).view(-1, 1)  # Flatten job states into a feature matrix
        data = Data(x=x, edge_index=edge_index)
        return data

    # Create edge connections between nodes in the graph (jobs and time steps)
    def create_edges(self):
        edges = []
        for job in range(self.J):
            for t in range(self.T - 1):
                edges.append((job * self.T + t, job * self.T + t + 1))  # Connect consecutive time steps
                edges.append((job * self.T + t + 1, job * self.T + t))  # Bidirectional edge
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Return edge tensor
        return edge_index

# Training function for the GNN model
def train_gnn(env, pn=None, mem=None, episodes=500, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128):
    n_actions = env.J * env.T  # Number of possible actions
    policy_net = GNN(in_channels=1, out_channels=n_actions) if pn is None else pn  # Initialize GNN policy net
    optimizer = optim.Adam(policy_net.parameters())  # Adam optimizer
    memory = deque(maxlen=10000) if mem is None else mem  # Experience replay memory
    episode_durations = []
    epsilon = eps_start

    for episode in range(episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        total_reward = 0
        while True:
            epsilon = max(epsilon * eps_decay, eps_end)  # Decay epsilon (for exploration)
            action = select_action_gnn(env, policy_net, epsilon)  # Select action using the policy network
            next_state, reward, done = env.step(action)  # Take a step in the environment
            total_reward += reward  # Accumulate reward
            memory.append(Experience(env.get_graph(), torch.tensor([[action]], dtype=torch.long), env.get_graph(), torch.tensor([reward], dtype=torch.float)))  # Store the experience in memory
            optimize_model_gnn(policy_net, optimizer, memory, gamma, batch_size)  # Update the model using experience replay
            if done:
                episode_durations.append(total_reward)  # Store the total reward for the episode
                break
    return policy_net, memory  # Return the trained policy and memory

# Select an action using epsilon-greedy strategy
def select_action_gnn(env, policy_net, epsilon):
    sample = random.random()  # Sample a random value
    if sample > epsilon:
        with torch.no_grad():
            state_graph = env.get_graph()  # Get current state as a graph
            q_values = policy_net(state_graph)  # Predict Q-values using the policy network
            return q_values.max(1)[1].item()  # Return the action with the highest Q-value
    else:
        return random.randrange(env.J * env.T)  # Random action for exploration

# Experience tuple for experience replay
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Update the policy network using experience replay and gradient descent
def optimize_model_gnn(policy_net, optimizer, memory, gamma, batch_size):
    if len(memory) < batch_size:
        return  # Skip optimization if not enough samples in memory
    experiences = random.sample(memory, batch_size)  # Sample random experiences from memory
    batch = Experience(*zip(*experiences))  # Unpack the batch
    
    state_batch = Batch.from_data_list([exp for exp in batch.state])  # Convert to graph batch
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = Batch.from_data_list([exp for exp in batch.next_state])

    state_action_values = policy_net(state_batch).gather(1, action_batch)  # Predict state-action values
    next_state_values = policy_net(next_state_batch).max(1)[0].detach()  # Predict next state values
    expected_state_action_values = (next_state_values * gamma) + reward_batch  # Compute target Q-values

    # Change from MSE loss to Huber loss (less sensitive to outliers)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate the trained GNN model
def evaluate_gnn_model(env, policy_net, episodes=1):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()  # Reset the environment
        while True:
            action = select_action_gnn(env, policy_net, epsilon=0)  # Select action greedily (epsilon = 0)
            state, reward, done = env.step(action)  # Take a step
            if done:
                break  # End the episode if done
        total_rewards += reward
        print(f"Episode {episode+1}: Reward: {reward}")  # Print reward for each episode
        print(state)
        print()
    average_reward = total_rewards / episodes
    print(f"Average Reward over {episodes} episodes: {average_reward}")  # Print the average reward

# Creating an instance for the ONTS problem with predefined parameters

u__job_priorities = np.array([3, 2, 1])
q__energy_consumption_per_job = np.array([1, 2, 1])
y_min_per_job = [1, 1, 1] 
y_max_per_job = [3, 4, 5]
t_min_per_job = [1, 1, 1]
t_max_per_job = [3, 4, 3]
p_min_per_job = [1, 1, 1]
p_max_per_job = [4, 5, 5]
w_min_per_job = [0, 0, 0]
w_max_per_job = [4, 5, 4]
r__energy_available_at_time_t = np.array([3, 3, 3, 3, 3])
gamma = 0.5
Vb = 1
Q = 10
p = 0.1
e = 0.9

env = ONTSEnv(u__job_priorities, q__energy_consumption_per_job, y_min_per_job, y_max_per_job, 
              t_min_per_job, t_max_per_job, p_min_per_job, p_max_per_job, w_min_per_job, 
              w_max_per_job, r__energy_available_at_time_t, gamma, Vb, Q, p, e)
env.reset()

# Initial training of the GNN policy
policy_net, memory = train_gnn(env, episodes=1000)

# Store GNN policy and memory into a binary file
with open('policy.txt', 'wb') as file:
    pickle.dump(policy_net, file)
with open('mem.txt', 'wb') as file:
    pickle.dump(memory, file)

# Evaluate the trained policy on the ONTS problem
evaluate_gnn_model(env, policy_net, episodes=10)
