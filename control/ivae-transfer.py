import os
import gym
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import trange
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict


problem = "Pendulum-v1"
env = gym.make(problem)
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

# IVAE
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, latent_dim):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z_e):
        distances = (z_e.pow(2).sum(dim=-1, keepdim=True) +
                     self.embedding.weight.pow(2).sum(dim=1) -
                     2 * torch.matmul(z_e, self.embedding.weight.t()))
        indices = torch.argmin(distances, dim=-1)
        return indices

    def straight_through(self, z_e, indices):
        z_q = self.embedding(indices)
        z_q_sg = z_e + (z_q - z_e).detach()
        return z_q, z_q_sg

class EncoderConv1(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(EncoderConv1, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.latent_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x, _ = torch.max(x, dim=2)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.latent_layer(x)
        return x

class DecoderMLP(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, hidden_dim, num_layers):
        super(DecoderMLP, self).__init__()
        self.input_layer = nn.Linear(latent_dim + condition_dim + input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, z, c, noise):
        x = torch.cat([z, c, noise], dim=1)
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)

class NoiseEstimator(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim):
        super(NoiseEstimator, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(latent_dim),  
            nn.Linear(latent_dim, hidden_dim),   
            nn.ReLU(),         
            nn.Dropout(0.5),   
            nn.Linear(hidden_dim, input_dim)   
        )

    def forward(self, x):
        return self.model(x)

class ConditionalVQVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim, num_embeddings, num_layers):
        super(ConditionalVQVAE, self).__init__()
        self.encoder = EncoderConv1(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = DecoderMLP(input_dim, latent_dim, condition_dim, hidden_dim, num_layers)
        self.vq = VectorQuantizer(num_embeddings, latent_dim)

    def forward(self, x, c):
        z_e = self.encoder(x)
        z_q = self.vq.straight_through(z_e)
        return self.decoder(z_q, c)

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 3
condition_dim = 4
latent_dim = 1
num_embeddings = 30 
beta = 1
alpha = 20
num_layers = 2
hidden_dim = 48
num_epochs = 300
learning_rate = 1e-4
data_type = 'pendulum'
ckp_path = "./checkpoint/"+ data_type + ".pth"
noise_path = "./checkpoint/"+ data_type + "_noise.pth"
csv_path = os.path.join("..", "control/dataset", data_type + ".csv")
pkl_path = os.path.join("..", "control/dataset", "data_offline_multiple.pkl")
save_path = os.path.join("..", "control/dataset", "data_ivae_multiple.pkl")
ckpmodel = torch.load(ckp_path)
ckpnoise = torch.load(noise_path)

model = ConditionalVQVAE(input_dim, condition_dim, hidden_dim, latent_dim, num_embeddings, num_layers).to(device)
model.load_state_dict(ckpmodel['model_state_dict'])
flow_model = NoiseEstimator(latent_dim, input_dim, hidden_dim).to(device)
flow_model.load_state_dict(ckpnoise['model_state_dict'])

def InvidualizedModel(states, labels):
    model.eval()
    with torch.no_grad():
        z_e = model.encoder(states)
        indices = model.vq(z_e)
        z_q, z_q_sg = model.vq.straight_through(z_e, indices)
        noise = flow_model(z_q_sg)
        output = model.decoder(z_q_sg, labels, noise)
    return z_q.cpu().squeeze().numpy(), output


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.latent_buffer = np.zeros((self.buffer_capacity, 1))
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.latent_buffer[index] = obs_tuple[0]
        self.state_buffer[index] = obs_tuple[1]
        self.action_buffer[index] = obs_tuple[2]
        self.reward_buffer[index] = obs_tuple[3]
        self.next_state_buffer[index] = obs_tuple[4]
        self.buffer_counter += 1

    def learn(self, actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer, gamma, tau):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        latent_batch = torch.FloatTensor(self.latent_buffer[batch_indices])
        state_batch = torch.FloatTensor(self.state_buffer[batch_indices])
        action_batch = torch.FloatTensor(self.action_buffer[batch_indices])
        reward_batch = torch.FloatTensor(self.reward_buffer[batch_indices])
        next_state_batch = torch.FloatTensor(self.next_state_buffer[batch_indices])

        with torch.no_grad():
            target_actions = target_actor(next_state_batch, latent_batch)
            target_value = target_critic(next_state_batch, target_actions)
            y = reward_batch + gamma * target_value

        critic_value = critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(critic_value, y)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actions = actor(state_batch, latent_batch)
        actor_loss = -critic(state_batch, actions).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        for param, target_param in zip(critic.parameters(), target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(actor.parameters(), target_actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_states+1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, state, latent):
        x = torch.cat([state, latent], dim=1)
        return self.layer(x) * upper_bound


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state_layer = nn.Sequential(
            nn.Linear(num_states, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.action_layer = nn.Sequential(
            nn.Linear(num_actions, 32),
            nn.ReLU()
        )
        self.concat_layer = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        state_out = self.state_layer(state)
        action_out = self.action_layer(action)
        concat = torch.cat([state_out, action_out], 1)
        return self.concat_layer(concat)

def policy(state, latent, noise_object, actor):
    latent = torch.tensor(latent, dtype=torch.float32)
    with torch.no_grad():
        action = actor(state, latent)
    noise = noise_object()
    legal_action = np.clip(action.numpy() + noise, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]

env = gym.make("Pendulum-v1",g=10.0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

with open(pkl_path, 'rb') as file:
    data = pickle.load(file)

extracted_data = [ ]
for item in data:
    converted_tuple = (item[0], item[3], str(item[1].tolist()).replace(',', ' '), item[2].tolist(), str(item[4].tolist()).replace(',', ' '))
    extracted_data.append(converted_tuple)

d1 = [row[0] for row in extracted_data]
d2 = [np.fromstring(row[2][1:-1], sep=' ') for row in extracted_data]
dcon = list(zip(d1, d2))
groups = defaultdict(list)
for key, value in dcon:
    groups[key].append(value)
states = []
for key, value in dcon:
    states.append(tuple(groups[key]))
states = torch.tensor(states).to(device).float() 
state = torch.tensor(np.array([np.fromstring(row[2][1:-1], sep=' ') for row in extracted_data])).to(device).float()
action = torch.tensor([row[3] for row in extracted_data]).to(device).float() 
labels = torch.cat([state, action], dim=1).to(device).float()
index, output = InvidualizedModel(states, labels)

samples = []
for idx, x in enumerate(data):
    latent = index[idx]
    state = x[1]
    action = x[2]
    reward = x[3]
    next_state = x[4]
    samples.append((latent, state, action, reward, next_state))

with open(save_path, 'wb') as file:
    pickle.dump(samples, file)

num_envs = 1
num_episodes = 100 
env = gym.make("Pendulum-v1", g=10.0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
MEMORY_SIZE = num_envs*num_episodes
memory = deque(maxlen=MEMORY_SIZE)

def collect_individual_samples(env, index, num_episodes):
    done = False
    state = env.reset()
    memory = []
    for _ in range(num_episodes):
        action = env.action_space.sample() 
        next_state, reward, done, _ = env.step(action)
        memory.append((index, state, action, reward, next_state, done))
        state = next_state
    return memory

for env_index in range(num_envs):
    gravity = 10
    random_samples = collect_individual_samples(env, env_index, num_episodes)
    for sample in random_samples:
        memory.append(sample)

with open(os.path.join("..", "control/dataset", "data_new_joiner.pkl"), 'wb') as file:
    pickle.dump(memory, file)

extracted_data = [ ]
for item in memory:
    converted_tuple = (item[0], item[3], str(item[1].tolist()).replace(',', ' '), item[2].tolist(), str(item[4].tolist()).replace(',', ' '))
    extracted_data.append(converted_tuple)

d1 = [row[0] for row in extracted_data]
d2 = [np.fromstring(row[2][1:-1], sep=' ') for row in extracted_data]
dcon = list(zip(d1, d2))
groups = defaultdict(list)
for key, value in dcon:
    groups[key].append(value)
states = []
for key, value in dcon:
    states.append(tuple(groups[key]))
states = torch.tensor(states).to(device).float()
state = torch.tensor(np.array([np.fromstring(row[2][1:-1], sep=' ') for row in extracted_data])).to(device).float()
action = torch.tensor([row[3] for row in extracted_data]).to(device).float() 
labels = torch.cat([state, action], dim=1).to(device).float()
index, output = InvidualizedModel(states, labels)

samples = []
for idx, x in enumerate(memory):
    latent = index[idx] 
    state = x[1] 
    action = x[2] 
    reward = x[3] 
    next_state = x[4] 
    samples.append((latent, state, action, reward, next_state))

with open(os.path.join("..", "control/dataset", "data_ivae_new_joiner.pkl"), 'wb') as file:
    pickle.dump(samples, file)

###################################################################
# Part 3: Transfer Trained DDPG to "Pendulum-v1"
###################################################################
env = gym.make("Pendulum-v1",g=10.0)
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

loaded_actor = Actor()
loaded_critic = Critic()
loaded_target_actor = Actor()
loaded_target_critic = Critic()
loaded_actor.load_state_dict(torch.load(os.path.join("..", "control/model_saving", "ivae_actor_model.pth")))
loaded_critic.load_state_dict(torch.load(os.path.join("..", "control/model_saving", "ivae_critic_model.pth")))
loaded_target_actor.load_state_dict(loaded_actor.state_dict())
loaded_target_critic.load_state_dict(loaded_critic.state_dict())

critic_lr = 0.002
actor_lr = 0.001
critic_optimizer = optim.Adam(loaded_critic.parameters(), lr=critic_lr)
actor_optimizer = optim.Adam(loaded_actor.parameters(), lr=actor_lr)

total_episodes = 100
gamma = 0.99
tau = 0.005
buffer = Buffer(50000, 64)

idx = np.array([index.mean()])
ep_reward_list = []
avg_reward_list = []

for ep in range(total_episodes):
    prev_state = env.reset()
    episodic_reward = 0

    while True:
        prev_state_torch = torch.FloatTensor([prev_state])
        action = policy(prev_state_torch, np.array([idx]), ou_noise, loaded_actor)
        state, reward, done, _ = env.step(action)

        buffer.record((index.mean(), prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn(loaded_actor, loaded_critic, loaded_target_actor, loaded_target_critic, actor_optimizer, critic_optimizer, gamma, tau)

        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

with open(os.path.join("..", "control/dataset", "reward_transfer_ivae.pkl"), 'wb') as file:
    pickle.dump(avg_reward_list, file)
