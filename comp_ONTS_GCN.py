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

# Este código importa todas as bibliotecas necessárias para a implementação.
# As bibliotecas principais incluem PyTorch, usada para aprendizado profundo, 
# e PyTorch Geometric (PyG) para redes neurais em grafos (GNN).
# São importados também otimizadores, amostragem e memória de experiência para facilitar o aprendizado por reforço (RL).

# Define a Rede Neural Gráfica (GNN) para aprendizado de política.
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        # Três camadas de convolução em grafos (GCN)
        self.conv1 = GCNConv(in_channels, 128)  # Primeira camada GCN
        self.conv2 = GCNConv(128, 128)          # Segunda camada GCN
        self.conv3 = GCNConv(128, 64)           # Terceira camada GCN
        # Camadas totalmente conectadas para processar a saída do GCN
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, out_channels)

    # Função de passagem para frente (forward) através da rede
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Aplica a primeira convolução em grafo, seguida por ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Aplica a segunda camada GCN com ReLU
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Aplica a terceira camada GCN com ReLU
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        # Pooling global médio no grafo para condensar a informação dos nós
        x = global_mean_pool(x, batch)
        # Passa pelas camadas totalmente conectadas para gerar a saída (espaço de ações)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# A classe ONTSEnv simula o ambiente do problema ONTS (escalonamento de tarefas).
class ONTSEnv:
    def __init__(self, u__job_priorities, q__energy_consumption_per_job, y_min_per_job, y_max_per_job, 
                 t_min_per_job, t_max_per_job, p_min_per_job, p_max_per_job, 
                 w_min_per_job, w_max_per_job, r__energy_available_at_time_t, gamma, Vb, Q, p, e, max_steps=None):
        
        # Inicializa as variáveis do ambiente e as restrições
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
        self.SoC_t = self.p  # Inicializa o estado de carga

        self.J, self.T = len(u__job_priorities), len(r__energy_available_at_time_t)
        self.max_steps = max_steps if max_steps is not None else self.T  # Define o número máximo de passos
        self.x__state = None
        self.phi__state = None
        self.steps_taken = 0
        self.reset()  # Inicializa o ambiente

    # Função reset para reiniciar o ambiente em um estado inicial
    def reset(self):
        # Reinicia as matrizes de estado para os trabalhos e variáveis auxiliares
        self.x__state = np.zeros((self.J, self.T), dtype=int)
        self.phi__state = np.zeros((self.J, self.T), dtype=int)
        self.steps_taken = 0
        self.SoC_t = self.p  # Reinicia o estado de carga
        return self.x__state.flatten()

    # Função step para realizar uma ação, atualizar o ambiente e retornar a recompensa e o próximo estado
    def step(self, action):
        job, time_step = divmod(action, self.T)  # Converte a ação em trabalho e etapa de tempo
        self.steps_taken += 1
        self.x__state[job, time_step] = 1 - self.x__state[job, time_step]  # Alterna o estado do trabalho na etapa de tempo
        self.build_phi_matrix()  # Atualiza a matriz auxiliar para as restrições de agendamento de tarefas
        reward, energy_exceeded = self.calculate_reward()  # Calcula a recompensa
        done = energy_exceeded or self.steps_taken >= self.max_steps  # Verifica se o episódio terminou
        return self.x__state, reward, done

    # Constrói a matriz auxiliar para acompanhar as ativações e desativações dos trabalhos
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

    # Verifica as restrições de energia para o agendamento dos trabalhos
    def check_energy_constraints(self):
        for t in range(self.T):
            totalEnergyRequiredAtTimeStep_t = 0
            for j in range(self.J):
                totalEnergyRequiredAtTimeStep_t += self.x__state[j][t] * self.q__energy_consumption_per_job[j]
            # Verifica se a energia total excede a energia disponível
            if totalEnergyRequiredAtTimeStep_t > self.r__energy_available_at_time_t[t] + (self.gamma * self.Vb):
                return -1, False  # Penaliza por exceder a energia
            # Calcula o estado de carga (SoC)
            exceedingPower = self.r__energy_available_at_time_t[t] - totalEnergyRequiredAtTimeStep_t
            i_t = exceedingPower / self.Vb
            self.SoC_t = self.SoC_t + (i_t * self.e) / (60 * self.Q)
            if self.SoC_t > 1:
                return -1, False  # Penaliza por exceder o SoC máximo
        return 0, False  # Sem violações de restrições

    # Verifica restrições específicas dos trabalhos como tempos mínimos e máximos, períodos, etc.
    def check_job_constraints(self):
        acc_reward = 0
        for j in range(self.J):
            # Verifica se os trabalhos são ativados nos intervalos corretos (w_min, w_max)
            for tw in range(self.w_min_per_job[j]):
                if self.x__state[j, tw] == 1:
                    acc_reward -= 1  # Penaliza por ativar em momento inadequado
            for tw in range(self.w_max_per_job[j], self.T):
                if self.x__state[j, tw] == 1:
                    acc_reward -= 1  # Penaliza por ativar em momento inadequado
                
            # Verifica os tempos mínimos e máximos que o trabalho pode estar ativo (y_min, y_max)
            sum_l = 0
            for t in range(self.T):
                sum_l += self.phi__state[j, t]
            if sum_l < self.y_min_per_job[j]:
                acc_reward -= 1  # Penaliza por não atingir o tempo mínimo de execução
            if sum_l > self.y_max_per_job[j]:
                acc_reward -= 1  # Penaliza por exceder o tempo máximo de execução
            
            # Verifica restrições de execução contínua (t_min, t_max)
            for t in range(self.T - self.t_min_per_job[j] + 1):
                tt_sum = 0
                for tt in range(t, t + self.t_min_per_job[j]):
                    tt_sum += self.x__state[j, tt]
                if tt_sum < self.t_min_per_job[j] * self.phi__state[j, t]:
                    acc_reward -= 1  # Penaliza por não atingir execução contínua mínima

            for t in range(self.T - self.t_max_per_job[j]):
                tt_sum = 0
                for tt in range(t, t + self.t_max_per_job[j] + 1):
                    tt_sum += self.x__state[j, tt]
                if tt_sum > self.t_max_per_job[j]:
                    acc_reward -= 1  # Penaliza por exceder a execução contínua máxima
                
            # Verifica restrições de execução periódica (p_min, p_max)
            for t in range(self.T - self.p_min_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_min_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l > 1:
                    acc_reward -= 1  # Penaliza por não cumprir a execução periódica mínima

            for t in range(self.T - self.p_max_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_max_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l < 1:
                    acc_reward -= 1  # Penaliza por exceder a execução periódica máxima
        return acc_reward

    # Calcula a recompensa com base nas restrições de energia e de trabalho
    def calculate_reward(self):
        rewardSum = 0
        reward, done = self.check_energy_constraints()  # Verifica primeiro as restrições de energia
        rewardSum += reward        
        if not done:
            reward = self.check_job_constraints()  # Verifica as restrições de trabalho a seguir
            rewardSum += reward
            for j in range(self.J):
                for t in range(self.T):
                    # A recompensa é dada com base nas prioridades dos trabalhos e consumo de energia,
                    # e é positiva apenas se todas as restrições forem atendidas.
                    if rewardSum == 0:
                        rewardSum += 10 * (self.u__job_priorities[j] * self.x__state[j, t]) * (self.r__energy_available_at_time_t[t] - self.q__energy_consumption_per_job[j])
            return rewardSum, False  # Continua se não houver violação
        return rewardSum, done  # Retorna a recompensa e o status de término

    # Obtém a representação em grafo do ambiente (usado pela GNN)
    def get_graph(self):
        edge_index = self.create_edges()  # Cria conexões entre os nós do grafo
        x = torch.tensor(self.x__state.flatten(), dtype=torch.float).view(-1, 1)  # Achata o estado do trabalho em uma matriz de características
        data = Data(x=x, edge_index=edge_index)
        return data

    # Cria conexões entre os nós no grafo (trabalhos e etapas de tempo)
    def create_edges(self):
        edges = []
        for job in range(self.J):
            for t in range(self.T - 1):
                edges.append((job * self.T + t, job * self.T + t + 1))  # Conecta etapas de tempo consecutivas
                edges.append((job * self.T + t + 1, job * self.T + t))  # Conexão bidirecional
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Retorna o tensor de arestas
        return edge_index

# Função de treinamento para o modelo GNN
def train_gnn(env, pn=None, mem=None, episodes=500, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128):
    n_actions = env.J * env.T  # Número de ações possíveis
    policy_net = GNN(in_channels=1, out_channels=n_actions) if pn is None else pn  # Inicializa a rede de política GNN
    optimizer = optim.Adam(policy_net.parameters())  # Otimizador Adam
    memory = deque(maxlen=10000) if mem is None else mem  # Memória de replay de experiência
    episode_durations = []
    epsilon = eps_start

    for episode in range(episodes):
        state = env.reset()  # Reinicia o ambiente no início de cada episódio
        total_reward = 0
        while True:
            epsilon = max(epsilon * eps_decay, eps_end)  # Decaimento do epsilon (para exploração)
            action = select_action_gnn(env, policy_net, epsilon)  # Seleciona ação usando a rede de política
            next_state, reward, done = env.step(action)  # Executa uma ação no ambiente
            total_reward += reward  # Acumula a recompensa
            memory.append(Experience(env.get_graph(), torch.tensor([[action]], dtype=torch.long), env.get_graph(), torch.tensor([reward], dtype=torch.float)))  # Armazena a experiência na memória
            optimize_model_gnn(policy_net, optimizer, memory, gamma, batch_size)  # Atualiza o modelo usando replay de experiência
            if done:
                episode_durations.append(total_reward)  # Armazena a recompensa total para o episódio
                break
    return policy_net, memory  # Retorna a política treinada e a memória

# Seleciona uma ação usando a estratégia epsilon-greedy
def select_action_gnn(env, policy_net, epsilon):
    sample = random.random()  # Gera um valor aleatório
    if sample > epsilon:
        with torch.no_grad():
            state_graph = env.get_graph()  # Obtém o estado atual como um grafo
            q_values = policy_net(state_graph)  # Prediz os valores Q usando a rede de política
            return q_values.max(1)[1].item()  # Retorna a ação com o maior valor Q
    else:
        return random.randrange(env.J * env.T)  # Ação aleatória para exploração

# Define a tupla de experiência para o replay
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Atualiza a rede de política usando replay de experiência e gradiente descendente
def optimize_model_gnn(policy_net, optimizer, memory, gamma, batch_size):
    if len(memory) < batch_size:
        return  # Pula a otimização se não houver amostras suficientes na memória
    experiences = random.sample(memory, batch_size)  # Amostra experiências aleatórias da memória
    batch = Experience(*zip(*experiences))  # Desempacota o lote de experiências
    
    state_batch = Batch.from_data_list([exp for exp in batch.state])  # Converte para lote de grafos
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = Batch.from_data_list([exp for exp in batch.next_state])

    state_action_values = policy_net(state_batch).gather(1, action_batch)  # Prediz os valores estado-ação
    next_state_values = policy_net(next_state_batch).max(1)[0].detach()  # Prediz os valores dos próximos estados
    expected_state_action_values = (next_state_values * gamma) + reward_batch  # Calcula os valores Q alvo

    # Mudança de perda MSE para Huber (menos sensível a outliers)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Avalia o modelo GNN treinado
def evaluate_gnn_model(env, policy_net, episodes=1):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()  # Reinicia o ambiente
        while True:
            action = select_action_gnn(env, policy_net, epsilon=0)  # Seleciona ação de forma gulosa (epsilon = 0)
            state, reward, done = env.step(action)  # Executa a ação
            if done:
                break  # Termina o episódio se estiver concluído
        total_rewards += reward
        print(f"Episode {episode+1}: Reward: {reward}")  # Imprime a recompensa de cada episódio
        print(state)
        print()
    average_reward = total_rewards / episodes
    print(f"Average Reward over {episodes} episodes: {average_reward}")  # Imprime a recompensa média
    return average_reward

# Cria uma instância do problema ONTS com parâmetros predefinidos

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

sum = 0
for _ in range(10):
    # Treinamento inicial da política GNN
    policy_net, memory = train_gnn(env, episodes=2000)

    # Avalia a política treinada no problema ONTS
    avg_rew = evaluate_gnn_model(env, policy_net, episodes=10)
    sum += avg_rew

print()
print(sum / 10)
