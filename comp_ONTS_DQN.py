import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from collections import namedtuple, deque
import pickle
import torch.nn as nn

# Import necessary libraries:
# - `torch` and `torch.nn` for defining and using neural networks
# - `numpy` for numerical computations
# - `random` for random action selection (exploration)
# - `pickle` for saving/loading trained models
# - `deque` for experience replay memory
# - `namedtuple` to structure experiences in replay memory

# Define the DQN Model
class DQN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(DQN, self).__init__()
        # Three fully connected layers for processing the state/action space
        self.fc1 = nn.Linear(n_inputs, 128)  # Input layer: n_inputs to 128 neurons
        self.fc2 = nn.Linear(128, 128)       # Second hidden layer: 128 neurons
        self.fc3 = nn.Linear(128, 64)        # Third hidden layer: 64 neurons
        self.out = nn.Linear(64, n_outputs)  # Output layer: 64 neurons to n_outputs (actions)

    def forward(self, x):
        # Forward pass through the network using ReLU activations for hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)  # Output Q-values for each action


# The ONTSEnv class simulates the environment for the Offline Nanosatellite Task Scheduling (ONTS) problem
class ONTSEnv:
    def __init__(self, u__job_priorities, q__energy_consumption_per_job, y_min_per_job, 
                 y_max_per_job, t_min_per_job, t_max_per_job, p_min_per_job, p_max_per_job,
                 w_min_per_job, w_max_per_job, r__energy_available_at_time_t, gamma, Vb, Q,
                 p, e, max_steps=None):
        # Initialization: Set job priorities, energy consumption, constraints, and other parameters
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
        self.SoC_t = self.p  # Initialize state of charge (SoC)

        # Determine the number of jobs (J) and time steps (T)
        self.J, self.T = len(u__job_priorities), len(r__energy_available_at_time_t)
        self.max_steps = max_steps if max_steps is not None else self.T  # Set max steps
        self.x__state = None  # State matrix for job activations
        self.phi__state = None  # Auxiliary matrix for constraints
        self.steps_taken = 0
        self.reset()  # Initialize the environment state
    
    # Reset the environment for a new episode
    def reset(self):
        # Reset state matrices and variables
        self.x__state = np.zeros((self.J, self.T), dtype=int)  # Job activations matrix
        self.phi__state = np.zeros((self.J, self.T), dtype=int)  # Auxiliary matrix
        self.steps_taken = 0
        self.SoC_t = self.p  # Reset state of charge
        return self.x__state.flatten()  # Return flattened state for DQN input
    
    # Step function to execute an action (activate/deactivate a job at a specific time step)
    def step(self, action):
        job, time_step = divmod(action, self.T)  # Convert action into job and time step
        self.steps_taken += 1  # Increment the number of steps taken
        # Flip the activation state of the job at the given time step
        self.x__state[job, time_step] = 1 - self.x__state[job, time_step]
        self.build_phi_matrix()  # Update auxiliary matrix
        reward, energy_exceeded = self.calculate_reward()  # Calculate reward based on constraints
        done = energy_exceeded or self.steps_taken >= self.max_steps  # Check if episode is done
        return self.x__state.flatten(), reward, done  # Return next state, reward, and done flag
    
    # Build the auxiliary matrix to track job activation/deactivation patterns
    def build_phi_matrix(self):
        for j in range(self.J):
            for t in range(self.T):
                if t == 0:
                    if self.x__state[j, t] > self.phi__state[j, t]:
                        self.phi__state[j, t] = 1  # Initialize at first time step
                else:
                    if (self.x__state[j, t] - self.x__state[j, t-1]) > self.phi__state[j, t]:
                        self.phi__state[j, t] = 1  # Detect transitions
                    if self.phi__state[j, t] > (2 - self.x__state[j, t] - self.x__state[j, t-1]):
                        self.phi__state[j, t] = 0  # Update based on state changes
                if self.phi__state[j, t] > self.x__state[j, t]:
                    self.phi__state[j, t] = 0  # Ensure auxiliary matrix respects job state

    # Check energy constraints (available energy vs. consumption)
    def check_energy_constraints(self):
        for t in range(self.T):
            totalEnergyRequiredAtTimeStep_t = 0
            # Calculate total energy consumption at time step `t`
            for j in range(self.J):
                totalEnergyRequiredAtTimeStep_t += self.x__state[j][t] * self.q__energy_consumption_per_job[j]
            # Check if total energy required exceeds the available energy
            if totalEnergyRequiredAtTimeStep_t > self.r__energy_available_at_time_t[t] + (self.gamma * self.Vb):
                return -1, False  # Penalize for exceeding available energy
            # Calculate new state of charge (SoC)
            exceedingPower = self.r__energy_available_at_time_t[t] - totalEnergyRequiredAtTimeStep_t
            i_t = exceedingPower / self.Vb
            self.SoC_t = self.SoC_t + (i_t * self.e) / (60 * self.Q)  # Update SoC
            if self.SoC_t > 1:
                return -1, False  # Penalize for exceeding SoC limit
        return 0, False  # No energy constraint violations

    # Check job-related constraints (activation times, periodicity, etc.)
    def check_job_constraints(self):
        acc_reward = 0  # Accumulate reward/penalty
        for j in range(self.J):
            # Check if jobs are activated within allowed time intervals (w_min, w_max)
            for tw in range(self.w_min_per_job[j]):
                if self.x__state[j, tw] == 1:
                    acc_reward -= 1  # Penalize for activation before allowed time
            for tw in range(self.w_max_per_job[j], self.T):
                if self.x__state[j, tw] == 1:
                    acc_reward -= 1  # Penalize for activation after allowed time
                
            # Check if jobs respect minimum and maximum activations (y_min, y_max)
            sum_l = 0
            for t in range(self.T):
                sum_l += self.phi__state[j, t]
            if sum_l < self.y_min_per_job[j]:
                acc_reward -= 1   # Penalize for not meeting minimum activations
            if sum_l > self.y_max_per_job[j]:
                acc_reward -= 1   # Penalize for exceeding maximum activations
            
            # Check if jobs respect continuous execution constraints (t_min, t_max)
            for t in range(self.T - self.t_min_per_job[j] + 1):
                tt_sum = 0
                for tt in range(t, t + self.t_min_per_job[j]):
                    tt_sum += self.x__state[j, tt] 
                if tt_sum < self.t_min_per_job[j] * self.phi__state[j, t]:
                    acc_reward -= 1  # Penalize for not meeting minimum continuous execution
            
            for t in range(self.T - self.t_max_per_job[j]):
                tt_sum = 0
                for tt in range(t, t + self.t_max_per_job[j] + 1):
                    tt_sum += self.x__state[j, tt]
                if tt_sum > self.t_max_per_job[j]:
                    acc_reward -= 1  # Penalize for exceeding maximum continuous execution

            # Check if jobs respect periodic execution constraints (p_min, p_max)
            for t in range(self.T - self.p_min_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_min_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l > 1:
                    acc_reward -= 1  # Penalize for failing periodicity constraints
                
            for t in range(self.T - self.p_max_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_max_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l < 1:
                    acc_reward -= 1  # Penalize for exceeding periodicity constraints
        
        return acc_reward  # Return accumulated reward/penalty
    
    # Calculate the total reward based on energy and job constraints
    def calculate_reward(self):
        rewardSum = 0  # Initialize reward sum
        reward, done = self.check_energy_constraints()  # First check energy constraints
        rewardSum += reward        
        if not done:
            reward = self.check_job_constraints()  # Then check job constraints
            rewardSum += reward
            for j in range(self.J):
                for t in range(self.T):
                    # Positive reward if all constraints are met, proportional to job priority and energy balance
                    if rewardSum == 0:
                        rewardSum += 10 * (self.u__job_priorities[j] * self.x__state[j, t]) * (self.r__energy_available_at_time_t[t] - self.q__energy_consumption_per_job[j])
            return rewardSum, False  # Return reward and continue (not done)
        return rewardSum, done  # Return reward and done flag


# Double DQN Training Function
def train_dqn(env, policy_net=None, target_net=None, episodes=500, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128, target_update=10):
    n_actions = env.J * env.T  # Number of possible actions
    n_inputs = env.J * env.T  # Size of the input (flattened state)

    # Initialize policy and target networks (if not provided)
    policy_net = DQN(n_inputs, n_actions) if policy_net is None else policy_net
    target_net = DQN(n_inputs, n_actions) if target_net is None else target_net
    target_net.load_state_dict(policy_net.state_dict())  # Copy weights from policy network to target network
    
    optimizer = optim.Adam(policy_net.parameters())  # Use Adam optimizer
    memory = deque(maxlen=10000)  # Initialize experience replay memory
    episode_durations = []
    epsilon = eps_start  # Initialize epsilon for exploration-exploitation balance

    for episode in range(episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        total_reward = 0
        while True:
            epsilon = max(epsilon * eps_decay, eps_end)  # Decrease epsilon (decaying exploration rate)
            action = select_action_dqn(env, policy_net, state, epsilon)  # Select action using epsilon-greedy strategy
            next_state, reward, done = env.step(action)  # Execute the action
            total_reward += reward  # Accumulate reward
            # Store the experience in memory (state, action, next_state, reward)
            memory.append(Experience(torch.tensor([state], dtype=torch.float), torch.tensor([[action]], dtype=torch.long), torch.tensor([next_state], dtype=torch.float), torch.tensor([reward], dtype=torch.float)))
            # Optimize the model using Double DQN approach
            optimize_model_dqn(policy_net, target_net, optimizer, memory, gamma, batch_size)
            state = next_state  # Update current state
            if done:
                episode_durations.append(total_reward)  # Store total reward for this episode
                break
        
        # Update the target network every `target_update` episodes
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())  # Sync target network with policy network

    return policy_net, target_net  # Return the trained policy and target networks


# Optimize the Double DQN model using experience replay and gradient descent
def optimize_model_dqn(policy_net, target_net, optimizer, memory, gamma, batch_size):
    if len(memory) < batch_size:
        return  # Skip optimization if there are not enough experiences
    experiences = random.sample(memory, batch_size)  # Sample random experiences from memory
    batch = Experience(*zip(*experiences))  # Unpack experiences

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    
    # Compute Q-values of the current state using the policy network
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute Q-values of the next state using the target network
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    
    # Calculate the expected Q-values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Use Huber loss (smooth_l1_loss) to handle outliers
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the network weights


# Select action for Double DQN using epsilon-greedy strategy
def select_action_dqn(env, policy_net, state, epsilon):
    sample = random.random()  # Random value for epsilon-greedy selection
    if sample > epsilon:
        with torch.no_grad():
            # Use the policy network to select the action with the highest Q-value
            return policy_net(torch.tensor([state], dtype=torch.float)).max(1)[1].view(1, 1).item()
    else:
        return random.randrange(env.J * env.T)  # Random action for exploration


# Evaluate the trained Double DQN model by running it on the environment
def evaluate_dqn_model(env, policy_net, episodes=1):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        while True:
            action = select_action_dqn(env, policy_net, state, epsilon=0)  # Select actions greedily (epsilon=0)
            state, reward, done = env.step(action)  # Execute action
            if done:
                break  # End episode if done
        total_rewards += reward  # Accumulate total rewards
        print(f"Episode {episode+1}: Reward: {reward}")  # Print reward for this episode
        print(state)
        print()
    average_reward = total_rewards / episodes
    print(f"Average Reward over {episodes} episodes: {average_reward}")  # Print average reward
    return average_reward  # Return the average reward


# Experience tuple to structure replay memory entries
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Define the ONTSEnv instance with predefined parameters for the ONTS problem
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

# Create environment instance
env = ONTSEnv(u__job_priorities, q__energy_consumption_per_job, y_min_per_job, y_max_per_job, 
              t_min_per_job, t_max_per_job, p_min_per_job, p_max_per_job, 
              w_min_per_job, w_max_per_job, r__energy_available_at_time_t, gamma, Vb, Q, p, e)
env.reset()

# Train Double DQN agent
policy_net, target_net = train_dqn(env, episodes=1000)

# Store trained DQN policy and target networks into binary files
with open('policy_dqn.pkl', 'wb') as file:
    pickle.dump(policy_net, file)
with open('target_dqn.pkl', 'wb') as file:
    pickle.dump(target_net, file)

# Evaluate the trained Double DQN model
evaluate_dqn_model(env, policy_net, episodes=10)
