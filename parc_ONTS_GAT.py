import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import torch.optim as optim
import random
from collections import namedtuple, deque

# Definição do modelo GAT (Graph Attention Network)
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        # Definindo duas camadas GATConv com cabeças de atenção
        self.conv1 = GATConv(in_channels, 128, heads=8, concat=True)
        self.conv2 = GATConv(128 * 8, 64, heads=8, concat=True)
        # Camadas totalmente conectadas (fully connected)
        self.fc1 = torch.nn.Linear(64 * 8, 32)
        self.fc2 = torch.nn.Linear(32, out_channels)

    def forward(self, data):
        # O gráfico é composto por nós (x), arestas (edge_index) e batch (para batch processing)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Passa os dados pelas camadas GAT e aplica a função de ativação ELU
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        # Aplica o pooling global (média) sobre os nós do gráfico
        x = global_mean_pool(x, batch)
        # Passa pelas camadas totalmente conectadas
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        return x

# Classe que define o ambiente ONTS (problema de otimização de agendamento de tarefas com consumo de energia)
class ONTSEnv:
    def __init__(self, job_priorities, energy_consumption, max_energy, max_steps=None):
        # Inicializa as prioridades das tarefas, o consumo de energia e outros parâmetros
        self.job_priorities = job_priorities
        self.energy_consumption = energy_consumption
        self.max_energy = max_energy
        self.J, self.T = energy_consumption.shape
        self.max_steps = max_steps if max_steps is not None else self.T
        self.state = None
        self.steps_taken = 0
        self.reset()
    
    def reset(self):
        # Reseta o ambiente e o estado (todas as tarefas não alocadas)
        self.state = np.zeros((self.J, self.T), dtype=int)
        self.steps_taken = 0
        return self.state.flatten()
    
    def step(self, action):
        # Atualiza o estado de acordo com a ação escolhida (toggle na alocação de uma tarefa em um tempo)
        job, time_step = divmod(action, self.T)
        self.steps_taken += 1
        self.state[job, time_step] = 1 - self.state[job, time_step]
        # Calcula a recompensa e verifica se a energia foi excedida
        reward, energy_exceeded = self.calculate_reward()
        done = energy_exceeded or self.steps_taken >= self.max_steps
        return self.state.flatten(), reward, done
    
    def calculate_reward(self):
        # Verifica se a energia máxima foi excedida em algum time step
        for t in range(self.T):
            totalEnergyAtTimeStep_t = 0
            for j in range(self.J):
                totalEnergyAtTimeStep_t += self.state[j][t] * self.energy_consumption[j][t]
            if totalEnergyAtTimeStep_t > self.max_energy:
                return -100, True  # Penaliza fortemente se a energia for excedida
        
        rewardSum = 0
        # Calcula a recompensa com base na prioridade das tarefas e no consumo de energia
        for j in range(self.J):
            for t in range(self.T):
                rewardSum += (self.job_priorities[j] * self.state[j][t]) * (self.max_energy + 1 - self.energy_consumption[j][t])

        # Verifica se todas as tarefas foram alocadas uma única vez
        if np.all(np.sum(self.state, axis=1) == 1):
            return rewardSum, False
        else:        
            return -1, False  # Penaliza levemente se nem todas as tarefas foram alocadas

    def get_graph(self):
        # Constrói o gráfico do estado atual para ser utilizado pelo modelo GAT
        edge_index = self.create_edges()
        x = torch.tensor(self.state.flatten(), dtype=torch.float).view(-1, 1)
        data = Data(x=x, edge_index=edge_index)
        return data

    def create_edges(self):
        # Cria as arestas entre os nós no gráfico (conectando time steps consecutivos para cada tarefa)
        edges = []
        for job in range(self.J):
            for t in range(self.T - 1):
                edges.append((job * self.T + t, job * self.T + t + 1))
                edges.append((job * self.T + t + 1, job * self.T + t))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

# Função de treinamento do Double GAT Agent
def train_gnn(env, pn=None, mem=None, episodes=500, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128, target_update=10):
    n_actions = env.J * env.T
    # Inicializa a rede de política (policy_net) e a rede alvo (target_net)
    policy_net = GAT(in_channels=1, out_channels=n_actions) if pn is None else pn
    target_net = GAT(in_channels=1, out_channels=n_actions)
    target_net.load_state_dict(policy_net.state_dict())  # Copia os pesos da rede de política para a rede alvo
    
    optimizer = optim.Adam(policy_net.parameters())
    memory = deque(maxlen=10000) if mem is None else mem
    episode_durations = []
    epsilon = eps_start

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            epsilon = max(epsilon * eps_decay, eps_end)
            action = select_action_gnn(env, policy_net, epsilon)
            next_state, reward, done = env.step(action)
            total_reward += reward
            # Armazena a experiência na memória
            memory.append(Experience(env.get_graph(), torch.tensor([[action]], dtype=torch.long), env.get_graph(), torch.tensor([reward], dtype=torch.float)))
            # Otimiza o modelo usando a rede de política e a rede alvo
            optimize_model_gnn(policy_net, target_net, optimizer, memory, gamma, batch_size)
            if done:
                episode_durations.append(total_reward)
                break
        
        # Atualiza a rede alvo a cada 'target_update' episódios
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())  # Atualiza os pesos da rede alvo para coincidir com a rede de política

    return policy_net, memory

# Função para selecionar a ação (estratégia epsilon-greedy)
def select_action_gnn(env, policy_net, epsilon):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            # Seleciona a ação baseada nas previsões da rede de política
            state_graph = env.get_graph()
            q_values = policy_net(state_graph)
            return q_values.max(1)[1].item()
    else:
        return random.randrange(env.J * env.T)

# Definição de experiência como um namedtuple (para armazenar transições)
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Função para otimizar o modelo Double GAT
def optimize_model_gnn(policy_net, target_net, optimizer, memory, gamma, batch_size):
    if len(memory) < batch_size:
        return
    # Amostra um batch de experiências da memória
    experiences = random.sample(memory, batch_size)
    batch = Experience(*zip(*experiences))
    
    state_batch = Batch.from_data_list([exp for exp in batch.state])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = Batch.from_data_list([exp for exp in batch.next_state])

    # Calcula os valores de Q da rede de política para o estado atual
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Calcula os valores de Q da rede alvo para o próximo estado
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    
    # Calcula os valores de Q esperados com base nas recompensas e no próximo estado
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Calcula a perda e realiza a retropropagação para otimizar a rede de política
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Função de avaliação do modelo GNN (GAT)
def evaluate_gnn_model(env, policy_net, episodes=1):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()
        while True:
            action = select_action_gnn(env, policy_net, epsilon=0)
            state, reward, done = env.step(action)
            if done:
                break
        total_rewards += reward
    average_reward = total_rewards / episodes
    print(f"Recompensa Média após {episodes} episódios: {average_reward}")
    return average_reward

# Configuração do ambiente e treinamento do modelo GAT
job_priorities = np.array([3, 2, 1])
energy_consumption = np.array([
    [2, 1, 3, 2, 1],
    [1, 2, 1, 3, 2],
    [2, 3, 2, 1, 1]
])
max_energy = 5

env = ONTSEnv(job_priorities, energy_consumption, max_energy)
env.reset()

# Treinamento inicial
policy_net, memory = train_gnn(env, episodes=2000)

# Avaliação do modelo treinado
evaluate_gnn_model(env, policy_net, episodes=10)
