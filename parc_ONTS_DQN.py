import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
import torch.nn.functional as F

# Definição do ambiente ONTS (Offline Nanosatellite Task Scheduling)
class ONTSEnv:
    def __init__(self, job_priorities, energy_consumption, max_energy, max_steps=None):
        # Inicializa o ambiente com as prioridades das tarefas, o consumo de energia e a energia máxima
        self.job_priorities = job_priorities
        self.energy_consumption = energy_consumption
        self.max_energy = max_energy
        self.J, self.T = energy_consumption.shape  # J = número de tarefas, T = número de intervalos de tempo
        self.max_steps = max_steps if max_steps is not None else self.T * self.J  # Máximo de passos
        self.state = None
        self.steps_taken = 0
        self.reset()  # Inicializa o ambiente
    
    def reset(self):
        # Reseta o ambiente, retornando o estado inicial
        self.state = np.zeros((self.J, self.T), dtype=int)  # Estado vazio (nenhuma tarefa agendada)
        self.steps_taken = 0  # Reinicia o contador de passos
        return self.state.flatten()  # Retorna o estado achatado (vetor 1D)

    def step(self, action):
        # Executa uma ação no ambiente: agenda/desagenda uma tarefa em um intervalo de tempo
        job, time_step = divmod(action, self.T)  # Mapeia a ação para uma tarefa e intervalo de tempo
        self.steps_taken += 1  # Incrementa o contador de passos
        
        # Alterna o estado da tarefa no intervalo de tempo (ativa se inativa, desativa se ativa)
        self.state[job, time_step] = 1 - self.state[job, time_step]
        
        # Calcula a recompensa e verifica se a energia máxima foi excedida
        reward, energy_exceeded = self.calculate_reward()
        
        # O episódio termina se a energia for excedida ou se todas as tarefas forem agendadas
        done = energy_exceeded or self.steps_taken >= self.max_steps or np.all(np.sum(self.state, axis=1) == 1)
        return self.state.flatten(), reward, done  # Retorna o novo estado, recompensa, e se o episódio acabou

    def calculate_reward(self):
        # Calcula a recompensa baseada na prioridade e no consumo de energia
        for t in range(self.T):
            totalEnergyAtTimeStep_t = 0
            for j in range(self.J):
                # Calcula o consumo de energia total no intervalo t
                totalEnergyAtTimeStep_t += self.state[j][t] * self.energy_consumption[j][t]
            
            # Penaliza se o consumo de energia exceder o limite
            if totalEnergyAtTimeStep_t > self.max_energy:
                return -100, True  # Penalidade por exceder o limite de energia
            
        rewardSum = 0
        for j in range(self.J):
            for t in range(self.T):
                # Recompensa diretamente proporcional à prioridade e inversamente proporcional ao consumo de energia
                rewardSum += (self.job_priorities[j] * self.state[j][t]) * (self.max_energy + 1 - self.energy_consumption[j][t])

        # Recompensa total se todas as tarefas forem agendadas exatamente uma vez
        if np.all(np.sum(self.state, axis=1) == 1):
            return rewardSum, False
        else:
            return -1, False  # Penalidade menor se nem todas as tarefas foram agendadas

# Estrutura para armazenar transições (experiência) na memória de replay
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Definição do modelo DQN (Deep Q-Network)
class DQN(nn.Module):
    def __init__(self, n_inputs):
        super(DQN, self).__init__()
        # Define a arquitetura da rede neural
        self.fc = nn.Sequential(
            nn.Linear(n_inputs, 128),  # Primeira camada oculta com 128 unidades
            nn.ReLU(),  # Função de ativação ReLU
            nn.Linear(128, 64),  # Segunda camada oculta com 64 unidades
            nn.ReLU(),  # Função de ativação ReLU
            nn.Linear(64, n_inputs)  # Camada de saída com 'n_inputs' (ações)
        )
    
    def forward(self, x):
        # Propaga os dados pela rede neural
        return self.fc(x)

# Definição do agente Double DQN
class DoubleDQNAgent:
    def __init__(self, n_inputs, n_actions):
        # Inicializa a rede de política e a rede alvo
        self.policy_net = DQN(n_inputs)
        self.target_net = DQN(n_inputs)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Sincroniza as redes
        self.optimizer = optim.Adam(self.policy_net.parameters())  # Otimizador Adam para a rede de política
        self.memory = deque(maxlen=10000)  # Memória de replay
        self.n_actions = n_actions
        self.batch_size = 128  # Tamanho do lote para otimização
        self.gamma = 0.99  # Fator de desconto
        self.epsilon_start = 1.0  # Valor inicial de epsilon (exploração)
        self.epsilon_end = 0.01  # Valor final de epsilon (exploração)
        self.epsilon_decay = 0.995  # Taxa de decaimento de epsilon
        self.epsilon = self.epsilon_start  # Valor inicial de epsilon

    def select_action(self, state):
        # Seleciona uma ação com a estratégia epsilon-greedy
        sample = random.random()
        if sample > self.epsilon:  # Exploração (ação com maior Q-valor)
            with torch.no_grad():
                return self.policy_net(torch.tensor([state], dtype=torch.float)).max(1)[1].view(1, 1).item()
        else:  # Exploração aleatória
            return random.randrange(self.n_actions)

    def optimize_model(self):
        # Otimiza o modelo usando amostras da memória de replay
        if len(self.memory) < self.batch_size:
            return
        # Amostra um lote de experiências da memória
        experiences = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*experiences))

        # Prepara os lotes de estados, ações, recompensas e próximos estados
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        
        # Calcula os Q-valores para os estados atuais e as ações
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Calcula os valores Q-alvo para os próximos estados
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Calcula a perda (loss) entre o Q-valor estimado e o Q-valor alvo
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Propaga o erro e atualiza os pesos da rede
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        # Atualiza o valor de epsilon para diminuir a exploração ao longo do tempo
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def update_target_net(self):
        # Atualiza a rede alvo copiando os pesos da rede de política
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Função de treinamento do agente
def train(agent, env, episodes=2000):
    for episode in range(episodes):
        # Inicializa o ambiente e a recompensa total por episódio
        state = env.reset()
        total_reward = 0
        while True:
            # O agente seleciona uma ação e executa no ambiente
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            # Armazena a transição na memória de replay
            agent.memory.append(Experience(torch.tensor([state], dtype=torch.float),
                                           torch.tensor([[action]], dtype=torch.long),
                                           torch.tensor([next_state], dtype=torch.float),
                                           torch.tensor([reward], dtype=torch.float)))
            state = next_state
            total_reward += reward
            # Otimiza o modelo em cada passo
            agent.optimize_model()
            if done:
                break
        # Atualiza epsilon após cada episódio
        agent.update_epsilon()
        # Atualiza a rede alvo a cada 10 episódios
        if episode % 10 == 0:
            agent.update_target_net()

# Definindo prioridades das tarefas, consumo de energia e energia máxima
job_priorities = np.array([3, 2, 1])
energy_consumption = np.array([
    [2, 1, 3, 2, 1],
    [1, 2, 1, 3, 2],
    [2, 3, 2, 1, 1]
])
max_energy = 5

# Função de avaliação do agente
def evaluate(agent, env, episodes=10):
    total_rewards = 0
    for episode in range(episodes):
        # Reseta o ambiente para cada episódio de avaliação
        state = env.reset()
        episode_reward = 0
        while True:
            # O agente seleciona uma ação e recebe a recompensa
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards += episode_reward  # Soma a recompensa por episódio
        
    # Calcula a recompensa média após todos os episódios de avaliação
    average_reward = total_rewards / episodes
    print(f"Average Reward over {episodes} episodes: {average_reward}")
    return average_reward

# Treinamento e avaliação do agente por 10 iterações
sum = 0
for _ in range(10):
    env = ONTSEnv(job_priorities, energy_consumption, max_energy)  # Inicializa o ambiente
    agent = DoubleDQNAgent(n_inputs=env.J * env.T, n_actions=env.J * env.T)  # Inicializa o agente Double DQN
    train(agent, env, episodes=2000)  # Treina o agente por 2000 episódios
    
    avg_rew = evaluate(agent, env, episodes=10)  # Avalia o agente em 10 episódios
    sum += avg_rew

# Exibe a recompensa média sobre todas as avaliações
print()
print(sum / 10)
