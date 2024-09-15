import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
import torch.nn.functional as F

# Implementação da Pointer Network (Rede de Ponteiro)
class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PointerNet, self).__init__()
        # Definindo o codificador e o decodificador LSTM
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)  # Codificador LSTM
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)  # Decodificador LSTM
        # Camada linear que gera os logits para a seleção de ações (pontos)
        self.pointer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Passa os dados pelo codificador e decodificador LSTM
        encoder_outputs, (hidden, cell) = self.encoder(x)
        decoder_outputs, _ = self.decoder(x, (hidden, cell))
        # Retorna os logits gerados pela camada de ponteiro
        pointer_logits = self.pointer(decoder_outputs)
        return pointer_logits

# Definição da estrutura de dados Experience para armazenar experiências (transições de estados)
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Definição do ambiente ONTSEnv (para agendamento de tarefas com consumo de energia)
class ONTSEnv:
    def __init__(self, job_priorities, energy_consumption, max_energy, max_steps=None):
        self.job_priorities = job_priorities  # Prioridade dos trabalhos
        self.energy_consumption = energy_consumption  # Consumo de energia de cada trabalho
        self.max_energy = max_energy  # Energia máxima permitida por etapa
        self.J, self.T = energy_consumption.shape  # J: número de trabalhos, T: número de etapas de tempo
        self.max_steps = max_steps if max_steps is not None else self.T  # Passos máximos
        self.state = None  # Estado atual do ambiente
        self.steps_taken = 0  # Número de passos já executados
        self.reset()  # Reseta o ambiente
    
    def reset(self):
        # Inicializa o estado para uma matriz de zeros e reinicia os passos
        self.state = np.zeros((self.J, self.T), dtype=int)
        self.steps_taken = 0
        return self.state.flatten()  # Retorna o estado como um vetor achatado
    
    def step(self, action):
        # Divide a ação em trabalho e etapa de tempo
        job, time_step = divmod(action, self.T)
        self.steps_taken += 1
        
        # Alterna o estado do trabalho (ativa/desativa)
        self.state[job, time_step] = 1 - self.state[job, time_step]
        
        # Calcula a recompensa e verifica se o consumo de energia foi excedido
        reward, energy_exceeded = self.calculate_reward()
        done = energy_exceeded or self.steps_taken >= self.max_steps
        return self.state.flatten(), reward, done
    
    def calculate_reward(self):
        # Calcula o consumo total de energia em cada etapa de tempo
        for t in range(self.T):
            totalEnergyAtTimeStep_t = 0
            for j in range(self.J):
                totalEnergyAtTimeStep_t += self.state[j][t] * self.energy_consumption[j][t]
            
            # Penaliza se o consumo de energia exceder o limite
            if totalEnergyAtTimeStep_t > self.max_energy:
                return -100, True  # Penalidade por exceder a energia
        
        # Calcula a recompensa com base nas prioridades e no consumo de energia
        rewardSum = 0
        for j in range(self.J):
            for t in range(self.T):
                rewardSum += (self.job_priorities[j] * self.state[j][t]) * (self.max_energy + 1 - self.energy_consumption[j][t])

        # Verifica se cada trabalho foi agendado exatamente uma vez
        if np.all(np.sum(self.state, axis=1) == 1):
            return rewardSum, False
        else:
            return -1, False  # Penaliza se nem todos os trabalhos foram agendados

# Função de treinamento da Double Pointer Network (Double PN)
def train_pointer_net(env, pn=None, mem=None, episodes=500, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128, target_update=10):
    # Número de ações no ambiente
    n_actions = env.J * env.T
    if pn is None:
        # Inicializa a rede de política (policy_net) e a rede alvo (target_net)
        policy_net = PointerNet(env.J * env.T, 128)
        target_net = PointerNet(env.J * env.T, 128)
        # Copia os pesos da rede de política para a rede alvo
        target_net.load_state_dict(policy_net.state_dict())
    else:
        policy_net = pn
        target_net = PointerNet(env.J * env.T, 128)
        target_net.load_state_dict(policy_net.state_dict())
    
    # Inicializa o otimizador Adam para a rede de política
    optimizer = optim.Adam(policy_net.parameters())
    if mem is None:
        # Memória para armazenar experiências (transições de estados)
        memory = deque(maxlen=10000)
    else:
        memory = mem
    episode_durations = []
    epsilon = eps_start  # Valor inicial de epsilon para exploração

    # Loop de episódios de treinamento
    for episode in range(episodes):
        state = env.reset()  # Reinicia o ambiente
        total_reward = 0
        t = 0
        while True:
            # Decaimento de epsilon para reduzir a exploração ao longo do tempo
            epsilon = max(epsilon * eps_decay, eps_end)
            # Seleciona a ação com base na rede de política (epsilon-greedy)
            action = select_action(state, policy_net, epsilon, n_actions)
            # Executa a ação no ambiente
            next_state, reward, done = env.step(action)
            
            total_reward += reward
            
            # Armazena a transição (experiência) na memória
            memory.append(Experience(torch.tensor([state], dtype=torch.float),
                                     torch.tensor([[action]], dtype=torch.long),
                                     torch.tensor([next_state], dtype=torch.float),
                                     torch.tensor([reward], dtype=torch.float)))
            
            # Atualiza o estado atual
            state = next_state

            # Otimiza a rede de política usando a rede alvo
            optimize_model(policy_net, target_net, optimizer, memory, gamma, batch_size)
            t += 1
            if done:
                episode_durations.append(t + 1)
                break

        # Atualiza a rede alvo a cada `target_update` episódios
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return policy_net, memory

# Função para selecionar a ação (usando epsilon-greedy)
def select_action(state, policy_net, epsilon, n_actions):
    sample = random.random()
    if sample > epsilon:
        # Usa a rede de política para selecionar a melhor ação (exploração)
        with torch.no_grad():
            state_tensor = torch.tensor([state], dtype=torch.float).unsqueeze(0)  # Adiciona uma dimensão de batch
            pointer_logits = policy_net(state_tensor)
            action_probs = F.softmax(pointer_logits.view(-1), dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            return action
    else:
        # Seleciona uma ação aleatória (exploração)
        return random.randrange(n_actions)

# Função de otimização do modelo (usando Double PN)
def optimize_model(policy_net, target_net, optimizer, memory, gamma, batch_size):
    if len(memory) < batch_size:
        return
    # Amostra um lote de experiências da memória
    experiences = random.sample(memory, batch_size)
    batch = Experience(*zip(*experiences))

    # Prepara os batches para os estados, ações, recompensas e próximos estados
    state_batch = torch.cat(batch.state).view(batch_size, 1, -1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state).view(batch_size, 1, -1)
    
    # Calcula os valores de ação no estado atual usando a rede de política
    state_action_values = policy_net(state_batch).view(batch_size, -1).gather(1, action_batch)
    
    # Calcula os valores do próximo estado usando a rede alvo (Double PN)
    next_state_values = target_net(next_state_batch).view(batch_size, -1).max(1)[0].detach()
    
    # Calcula o valor esperado das ações
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    # Calcula a perda (loss) e otimiza a rede de política
    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()  # Zera os gradientes
    loss.backward()  # Propaga os gradientes
    optimizer.step()  # Atualiza os pesos da rede

# Função de avaliação do modelo treinado
def evaluate_model(env, policy_net, episodes=1):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()  # Reinicia o ambiente
        while True:
            # Seleciona a ação usando a rede de política (sem exploração)
            action = select_action(state, policy_net, epsilon=0, n_actions=env.J * env.T)
            state, reward, done = env.step(action)
            if done:
                break
        total_rewards += reward
    # Calcula e imprime a recompensa média
    average_reward = total_rewards / episodes
    print(f"Recompensa Média em {episodes} episódios: {average_reward}")
    return average_reward

# Execução do treinamento
job_priorities = np.array([3, 2, 1])  # Prioridades dos trabalhos
energy_consumption = np.array([  # Consumo de energia dos trabalhos em cada etapa
    [2, 1, 3, 2, 1],
    [1, 2, 1, 3, 2],
    [2, 3, 2, 1, 1]
])
max_energy = 5  # Energia máxima permitida por etapa

sum = 0
# Treina e avalia o modelo por 10 execuções
for _ in range(10):
    env = ONTSEnv(job_priorities, energy_consumption, max_energy)
    env.reset()
    # Treinamento da Double Pointer Network
    policy_net, memory = train_pointer_net(env, episodes=2000)

    # Avaliação do modelo
    avg_rew = evaluate_model(env, policy_net, episodes=10)
    sum += avg_rew

# Imprime a recompensa média após 10 execuções
print()
print(sum / 10)
