import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import torch.optim as optim
import random
from collections import namedtuple, deque

# Definição do modelo GCN (Graph Convolutional Network)
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        # Definição de três camadas convolucionais gráficas
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 64)
        # Definição de duas camadas lineares
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, out_channels)

    def forward(self, data):
        # Forward pass pela rede neural gráfica
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        # Pooling global para consolidar os dados dos nós do grafo
        x = global_mean_pool(x, batch)
        # Passagem pelas camadas lineares
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# Classe de ambiente para o problema de otimização (ONTSEnv)
class ONTSEnv:
    def __init__(self, job_priorities, energy_consumption, max_energy, max_steps=None):
        # Inicialização dos parâmetros do ambiente
        self.job_priorities = job_priorities
        self.energy_consumption = energy_consumption
        self.max_energy = max_energy
        self.J, self.T = energy_consumption.shape
        self.max_steps = max_steps if max_steps is not None else self.T
        self.state = None
        self.steps_taken = 0
        self.reset()
    
    def reset(self):
        # Reinicializa o ambiente, com todos os trabalhos desativados
        self.state = np.zeros((self.J, self.T), dtype=int)
        self.steps_taken = 0
        return self.state.flatten()
    
    def step(self, action):
        # Realiza uma ação no ambiente, ativando ou desativando um trabalho em um tempo específico
        job, time_step = divmod(action, self.T)
        self.steps_taken += 1
        self.state[job, time_step] = 1 - self.state[job, time_step]
        reward, energy_exceeded = self.calculate_reward()
        done = energy_exceeded or self.steps_taken >= self.max_steps
        return self.state.flatten(), reward, done
    
    def calculate_reward(self):
        # Calcula a recompensa com base no consumo de energia e na prioridade dos trabalhos
        for t in range(self.T):
            totalEnergyAtTimeStep_t = 0
            for j in range(self.J):
                totalEnergyAtTimeStep_t += self.state[j][t] * self.energy_consumption[j][t]
            
            if totalEnergyAtTimeStep_t > self.max_energy:
                return -100, True  # Penaliza se a energia máxima for excedida
            
        rewardSum = 0
        for j in range(self.J):
            for t in range(self.T):
                # Recompensa proporcional à prioridade e inversamente proporcional ao consumo de energia
                rewardSum += (self.job_priorities[j] * self.state[j][t]) * (self.max_energy + 1 - self.energy_consumption[j][t])

        
        if np.all(np.sum(self.state, axis=1) == 1):  # Se todos os trabalhos forem agendados exatamente uma vez
            return rewardSum, False
        else:        
            return -1, False  # Penaliza se nem todos os trabalhos forem agendados

    def get_graph(self):
        # Gera um grafo com base no estado atual do ambiente
        edge_index = self.create_edges()
        x = torch.tensor(self.state.flatten(), dtype=torch.float).view(-1, 1)
        data = Data(x=x, edge_index=edge_index)
        return data

    def create_edges(self):
        # Cria as arestas para o grafo entre os tempos consecutivos de cada trabalho
        edges = []
        for job in range(self.J):
            for t in range(self.T - 1):
                edges.append((job * self.T + t, job * self.T + t + 1))
                edges.append((job * self.T + t + 1, job * self.T + t))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

# Função de treinamento para o agente Double GCN
def train_gnn(env, pn=None, mem=None, episodes=500, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128, target_update=10):
    n_actions = env.J * env.T
    # Inicializa as redes de política (policy) e de alvo (target)
    policy_net = GCN(in_channels=1, out_channels=n_actions) if pn is None else pn
    target_net = GCN(in_channels=1, out_channels=n_actions)
    target_net.load_state_dict(policy_net.state_dict())  # Copia os pesos da rede de política para a rede de alvo
    
    optimizer = optim.Adam(policy_net.parameters())
    memory = deque(maxlen=10000) if mem is None else mem
    episode_durations = []
    epsilon = eps_start

    for episode in range(episodes):
        # Reinicia o ambiente no início de cada episódio
        state = env.reset()
        total_reward = 0
        while True:
            epsilon = max(epsilon * eps_decay, eps_end)
            # Seleciona uma ação com base na política atual (rede GCN)
            action = select_action_gnn(env, policy_net, epsilon)
            next_state, reward, done = env.step(action)
            total_reward += reward
            # Armazena a experiência na memória
            memory.append(Experience(env.get_graph(), torch.tensor([[action]], dtype=torch.long), env.get_graph(), torch.tensor([reward], dtype=torch.float)))
            # Otimiza a rede de política usando a rede de alvo para calcular os valores futuros
            optimize_model_gnn(policy_net, target_net, optimizer, memory, gamma, batch_size)
            if done:
                episode_durations.append(total_reward)
                break
        
        # Atualiza a rede de alvo a cada `target_update` episódios
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return policy_net, memory

# Otimiza o modelo com a abordagem Double GCN
def optimize_model_gnn(policy_net, target_net, optimizer, memory, gamma, batch_size):
    if len(memory) < batch_size:
        return
    experiences = random.sample(memory, batch_size)
    batch = Experience(*zip(*experiences))
    
    # Cria batches de estados e ações para treinar a rede
    state_batch = Batch.from_data_list([exp for exp in batch.state])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = Batch.from_data_list([exp for exp in batch.next_state])

    # Calcula os valores de ação-estado usando a rede de política
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Calcula os valores do próximo estado usando a rede de alvo
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    
    # Calcula os valores esperados de estado-ação
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Calcula a perda e otimiza a rede de política
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Seleciona uma ação com base na rede de política
def select_action_gnn(env, policy_net, epsilon):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            state_graph = env.get_graph()
            q_values = policy_net(state_graph)
            return q_values.max(1)[1].item()
    else:
        return random.randrange(env.J * env.T)

# Namedtuple para armazenar as experiências (transições)
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Função de avaliação do modelo GNN
def evaluate_gnn_model(env, policy_net, episodes=1):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()
        while True:
            # Seleciona a ação usando a política sem exploração (epsilon = 0)
            action = select_action_gnn(env, policy_net, epsilon=0)
            state, reward, done = env.step(action)
            if done:
                break
        total_rewards += reward
    average_reward = total_rewards / episodes
    print(f"Recompensa Média em {episodes} episódios: {average_reward}")
    return average_reward

# Execução do Treinamento
job_priorities = np.array([3, 2, 1])
energy_consumption = np.array([
    [2, 1, 3, 2, 1],
    [1, 2, 1, 3, 2],
    [2, 3, 2, 1, 1]
])
max_energy = 5

sum = 0
# Treina e avalia o modelo por 10 execuções
for _ in range(10):
    env = ONTSEnv(job_priorities, energy_consumption, max_energy)
    env.reset()

    # Treinamento inicial
    policy_net, memory = train_gnn(env, episodes=2000)

    # Avaliação do modelo
    evaluate_gnn_model(env, policy_net, episodes=10)
