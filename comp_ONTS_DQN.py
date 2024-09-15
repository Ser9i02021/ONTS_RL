import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from collections import namedtuple, deque
import pickle
import torch.nn as nn

# Importar as bibliotecas necessárias:
# - `torch` e `torch.nn` para definir e usar redes neurais
# - `numpy` para cálculos numéricos
# - `random` para a seleção de ações aleatórias (exploração)
# - `pickle` para salvar/carregar modelos treinados
# - `deque` para memória de replay de experiência
# - `namedtuple` para estruturar experiências na memória de replay

# Definir o Modelo DQN
class DQN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(DQN, self).__init__()
        # Três camadas totalmente conectadas para processar o espaço de estados/ações
        self.fc1 = nn.Linear(n_inputs, 128)  # Camada de entrada: n_inputs para 128 neurônios
        self.fc2 = nn.Linear(128, 128)       # Segunda camada oculta: 128 neurônios
        self.fc3 = nn.Linear(128, 64)        # Terceira camada oculta: 64 neurônios
        self.out = nn.Linear(64, n_outputs)  # Camada de saída: 64 neurônios para n_outputs (ações)

    def forward(self, x):
        # Passagem direta pela rede usando funções de ativação ReLU nas camadas ocultas
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)  # Saída dos valores de Q para cada ação


# A classe ONTSEnv simula o ambiente para o problema de Agendamento de Tarefas de Nanosatélites (ONTS)
class ONTSEnv:
    def __init__(self, u__job_priorities, q__energy_consumption_per_job, y_min_per_job, 
                 y_max_per_job, t_min_per_job, t_max_per_job, p_min_per_job, p_max_per_job,
                 w_min_per_job, w_max_per_job, r__energy_available_at_time_t, gamma, Vb, Q,
                 p, e, max_steps=None):
        # Inicialização: definir as prioridades dos trabalhos, consumo de energia, restrições e outros parâmetros
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
        self.SoC_t = self.p  # Inicializar o estado de carga (SoC)

        # Determinar o número de trabalhos (J) e passos de tempo (T)
        self.J, self.T = len(u__job_priorities), len(r__energy_available_at_time_t)
        self.max_steps = max_steps if max_steps is not None else self.T  # Definir o número máximo de etapas
        self.x__state = None  # Matriz de estado para ativações de trabalhos
        self.phi__state = None  # Matriz auxiliar para verificar restrições
        self.steps_taken = 0
        self.reset()  # Inicializar o estado do ambiente
    
    # Função para reiniciar o ambiente no início de cada episódio
    def reset(self):
        # Redefinir as matrizes de estado e variáveis
        self.x__state = np.zeros((self.J, self.T), dtype=int)  # Matriz de ativações de trabalhos
        self.phi__state = np.zeros((self.J, self.T), dtype=int)  # Matriz auxiliar
        self.steps_taken = 0
        self.SoC_t = self.p  # Redefinir o estado de carga
        return self.x__state.flatten()  # Retornar o estado achatado para a entrada do DQN
    
    # Função de passo para executar uma ação (ativar/desativar um trabalho em um passo de tempo específico)
    def step(self, action):
        job, time_step = divmod(action, self.T)  # Converter ação em trabalho e passo de tempo
        self.steps_taken += 1  # Incrementar o número de passos realizados
        # Alternar o estado de ativação do trabalho no passo de tempo fornecido
        self.x__state[job, time_step] = 1 - self.x__state[job, time_step]
        self.build_phi_matrix()  # Atualizar a matriz auxiliar
        reward, energy_exceeded = self.calculate_reward()  # Calcular a recompensa com base nas restrições
        done = energy_exceeded or self.steps_taken >= self.max_steps  # Verificar se o episódio terminou
        return self.x__state.flatten(), reward, done  # Retornar o próximo estado, recompensa e flag de término
    
    # Construir a matriz auxiliar para rastrear padrões de ativação/desativação de trabalhos
    def build_phi_matrix(self):
        for j in range(self.J):
            for t in range(self.T):
                if t == 0:
                    if self.x__state[j, t] > self.phi__state[j, t]:
                        self.phi__state[j, t] = 1  # Inicializar no primeiro passo de tempo
                else:
                    if (self.x__state[j, t] - self.x__state[j, t-1]) > self.phi__state[j, t]:
                        self.phi__state[j, t] = 1  # Detectar transições
                    if self.phi__state[j, t] > (2 - self.x__state[j, t] - self.x__state[j, t-1]):
                        self.phi__state[j, t] = 0  # Atualizar com base nas mudanças de estado
                if self.phi__state[j, t] > self.x__state[j, t]:
                    self.phi__state[j, t] = 0  # Garantir que a matriz auxiliar respeite o estado do trabalho

    # Verificar restrições de energia (energia disponível vs. consumo)
    def check_energy_constraints(self):
        for t in range(self.T):
            totalEnergyRequiredAtTimeStep_t = 0
            # Calcular o consumo total de energia no passo de tempo `t`
            for j in range(self.J):
                totalEnergyRequiredAtTimeStep_t += self.x__state[j][t] * self.q__energy_consumption_per_job[j]
            # Verificar se a energia total requerida excede a energia disponível
            if totalEnergyRequiredAtTimeStep_t > self.r__energy_available_at_time_t[t] + (self.gamma * self.Vb):
                return -1, False  # Penalizar por exceder a energia disponível
            # Calcular o novo estado de carga (SoC)
            exceedingPower = self.r__energy_available_at_time_t[t] - totalEnergyRequiredAtTimeStep_t
            i_t = exceedingPower / self.Vb
            self.SoC_t = self.SoC_t + (i_t * self.e) / (60 * self.Q)  # Atualizar o SoC
            if self.SoC_t > 1:
                return -1, False  # Penalizar por exceder o limite de SoC
        return 0, False  # Nenhuma violação de restrição de energia

    # Verificar restrições relacionadas ao trabalho (tempos de ativação, periodicidade, etc.)
    def check_job_constraints(self):
        acc_reward = 0  # Acumular recompensa/penalidade
        for j in range(self.J):
            # Verificar se os trabalhos são ativados dentro dos intervalos permitidos (w_min, w_max)
            for tw in range(self.w_min_per_job[j]):
                if self.x__state[j, tw] == 1:
                    acc_reward -= 1  # Penalizar por ativação antes do tempo permitido
            for tw in range(self.w_max_per_job[j], self.T):
                if self.x__state[j, tw] == 1:
                    acc_reward -= 1  # Penalizar por ativação após o tempo permitido
                
            # Verificar se os trabalhos respeitam o número mínimo e máximo de ativações (y_min, y_max)
            sum_l = 0
            for t in range(self.T):
                sum_l += self.phi__state[j, t]
            if sum_l < self.y_min_per_job[j]:
                acc_reward -= 1   # Penalizar por não atingir o número mínimo de ativações
            if sum_l > self.y_max_per_job[j]:
                acc_reward -= 1   # Penalizar por exceder o número máximo de ativações
            
            # Verificar se os trabalhos respeitam as restrições de execução contínua (t_min, t_max)
            for t in range(self.T - self.t_min_per_job[j] + 1):
                tt_sum = 0
                for tt in range(t, t + self.t_min_per_job[j]):
                    tt_sum += self.x__state[j, tt] 
                if tt_sum < self.t_min_per_job[j] * self.phi__state[j, t]:
                    acc_reward -= 1  # Penalizar por não respeitar o mínimo de execução contínua
            
            for t in range(self.T - self.t_max_per_job[j]):
                tt_sum = 0
                for tt in range(t, t + self.t_max_per_job[j] + 1):
                    tt_sum += self.x__state[j, tt]
                if tt_sum > self.t_max_per_job[j]:
                    acc_reward -= 1  # Penalizar por exceder o máximo de execução contínua

            # Verificar se os trabalhos respeitam as restrições de execução periódica (p_min, p_max)
            for t in range(self.T - self.p_min_per_job[j] + 1):
                sum_l = 0
                for l in range(t, t + self.p_min_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l > 1:
                    acc_reward -= 1  # Penalizar por não respeitar a periodicidade mínima
                
            for t in range(self.T - self.p_max_per_job[j] + 1):
                sum_l = 0
                for l em range(t, t + self.p_max_per_job[j]):
                    sum_l += self.phi__state[j, l]
                if sum_l < 1:
                    acc_reward -= 1  # Penalizar por exceder a periodicidade máxima
        
        return acc_reward  # Retornar a recompensa acumulada/penalidade
    
    # Calcular a recompensa total com base nas restrições de energia e trabalho
    def calculate_reward(self):
        rewardSum = 0  # Inicializar a soma da recompensa
        reward, done = self.check_energy_constraints()  # Primeiro verificar as restrições de energia
        rewardSum += reward        
        if not done:
            reward = self.check_job_constraints()  # Em seguida, verificar as restrições dos trabalhos
            rewardSum += reward
            for j in range(self.J):
                for t in range(self.T):
                    # Recompensa positiva se todas as restrições forem respeitadas, proporcional à prioridade e energia restante
                    if rewardSum == 0:
                        rewardSum += 10 * (self.u__job_priorities[j] * self.x__state[j, t]) * (self.r__energy_available_at_time_t[t] - self.q__energy_consumption_per_job[j])
            return rewardSum, False  # Retornar recompensa e continuar (não terminado)
        return rewardSum, done  # Retornar recompensa e flag de término


# Função de Treinamento do Double DQN
def train_dqn(env, policy_net=None, target_net=None, episodes=500, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, batch_size=128, target_update=10):
    n_actions = env.J * env.T  # Número de ações possíveis
    n_inputs = env.J * env.T  # Tamanho da entrada (estado achatado)

    # Inicializar redes de política e alvo (se não fornecidas)
    policy_net = DQN(n_inputs, n_actions) if policy_net is None else policy_net
    target_net = DQN(n_inputs, n_actions) if target_net is None else target_net
    target_net.load_state_dict(policy_net.state_dict())  # Copiar pesos da rede de política para a rede alvo
    
    optimizer = optim.Adam(policy_net.parameters())  # Usar otimizador Adam
    memory = deque(maxlen=10000)  # Inicializar memória de replay de experiência
    episode_durations = []
    epsilon = eps_start  # Inicializar epsilon para o equilíbrio exploração-exploração

    for episode in range(episodes):
        state = env.reset()  # Reiniciar o ambiente no início de cada episódio
        total_reward = 0
        while True:
            epsilon = max(epsilon * eps_decay, eps_end)  # Diminuir epsilon (taxa decrescente de exploração)
            action = select_action_dqn(env, policy_net, state, epsilon)  # Selecionar ação usando estratégia epsilon-greedy
            next_state, reward, done = env.step(action)  # Executar a ação
            total_reward += reward  # Acumular recompensa
            # Armazenar a experiência na memória (estado, ação, próximo estado, recompensa)
            memory.append(Experience(torch.tensor([state], dtype=torch.float), torch.tensor([[action]], dtype=torch.long), torch.tensor([next_state], dtype=torch.float), torch.tensor([reward], dtype=torch.float)))
            # Otimizar o modelo usando o Double DQN
            optimize_model_dqn(policy_net, target_net, optimizer, memory, gamma, batch_size)
            state = next_state  # Atualizar o estado atual
            if done:
                episode_durations.append(total_reward)  # Armazenar a recompensa total deste episódio
                break
        
        # Atualizar a rede alvo a cada `target_update` episódios
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())  # Sincronizar a rede alvo com a rede de política

    return policy_net, target_net  # Retornar as redes treinadas de política e alvo


# Otimizar o modelo Double DQN usando replay de experiência e gradiente descendente
def optimize_model_dqn(policy_net, target_net, optimizer, memory, gamma, batch_size):
    if len(memory) < batch_size:
        return  # Pular otimização se não houver experiências suficientes
    experiences = random.sample(memory, batch_size)  # Amostrar experiências aleatórias da memória
    batch = Experience(*zip(*experiences))  # Descompactar experiências

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    
    # Calcular os valores de Q do estado atual usando a rede de política
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Calcular os valores de Q do próximo estado usando a rede alvo
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    
    # Calcular os valores de Q esperados
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Usar a perda de Huber (smooth_l1_loss) para lidar com outliers
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()  # Zerar os gradientes
    loss.backward()  # Retropropagar a perda
    optimizer.step()  # Atualizar os pesos da rede


# Selecionar ação para o Double DQN usando a estratégia epsilon-greedy
def select_action_dqn(env, policy_net, state, epsilon):
    sample = random.random()  # Valor aleatório para a seleção epsilon-greedy
    if sample > epsilon:
        with torch.no_grad():
            # Usar a rede de política para selecionar a ação com o maior valor de Q
            return policy_net(torch.tensor([state], dtype=torch.float)).max(1)[1].view(1, 1).item()
    else:
        return random.randrange(env.J * env.T)  # Ação aleatória para exploração


# Avaliar o modelo Double DQN treinado executando-o no ambiente
def evaluate_dqn_model(env, policy_net, episodes=1):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()  # Reiniciar o ambiente no início de cada episódio
        while True:
            action = select_action_dqn(env, policy_net, state, epsilon=0)  # Selecionar ações de forma gananciosa (epsilon=0)
            state, reward, done = env.step(action)  # Executar a ação
            if done:
                break  # Encerrar episódio se estiver feito
        total_rewards += reward  # Acumular recompensas totais
        print(f"Episódio {episode+1}: Recompensa: {reward}")  # Exibir a recompensa deste episódio
        print(state)
        print()
    average_reward = total_rewards / episodes
    print(f"Recompensa média nos {episodes} episódios: {average_reward}")  # Exibir a recompensa média
    return average_reward  # Retornar a recompensa média


# Tupla de experiência para estruturar entradas na memória de replay
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

# Definir a instância do ONTSEnv com parâmetros pré-definidos para o problema ONTS
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

# Criar instância do ambiente
env = ONTSEnv(u__job_priorities, q__energy_consumption_per_job, y_min_per_job, y_max_per_job, 
              t_min_per_job, t_max_per_job, p_min_per_job, p_max_per_job, 
              w_min_per_job, w_max_per_job, r__energy_available_at_time_t, gamma, Vb, Q, p, e)
env.reset()

# Treinar o agente Double DQN
policy_net, target_net = train_dqn(env, episodes=1000)

# Armazenar as redes de política e alvo treinadas em arquivos binários
with open('policy_dqn.pkl', 'wb') as file:
    pickle.dump(policy_net, file)
with open('target_dqn.pkl', 'wb') as file:
    pickle.dump(target_net, file)

# Avaliar o modelo Double DQN treinado
evaluate_dqn_model(env, policy_net, episodes=10)
