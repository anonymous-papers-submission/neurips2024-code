import json
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


def generate_mapping(original_list):
    mapping_dict = {}
    for index, item in enumerate(original_list):
        mapping_dict[item] = index + 1
    return mapping_dict

###############################################################################
# Data preparsion
###############################################################################
np.random.seed(42) 
torch.manual_seed(42)
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
s_prime_ids = torch.load('./DatasetFiltered/s_prime_ids.pth')
conditions_ids = torch.load('./DatasetFiltered/conditions_ids.pth')
embedding_ids = torch.load('./DatasetFiltered/embedding_ids.pth')
with open('./DatasetFiltered/label_ids.json', 'r') as f:
    label_ids = json.load(f)
csv_file_path = './DatasetFiltered/personality.csv'
df = pd.read_csv(csv_file_path)
indices_to_delete = ['dialog']
factors = [x for x in df.columns.tolist() if x not in indices_to_delete]

bs = 32
seqlen = 50
percentage = 0.8
sample_size = 2000
ckp_path = "./checkpoint/cvqvae_checkpoint.pth"
mapping_dict = generate_mapping(label_ids)
mapped_values = [mapping_dict.get(item, -1) for item in label_ids]
s_prime = torch.stack(s_prime_ids)[:sample_size,:,:,:].to(device)
conditions = torch.stack(conditions_ids)[:sample_size,:,:,:].to(device)
embeddings = torch.stack(embedding_ids)[:sample_size,:,:,:].to(device)
label_idx = torch.tensor(mapped_values)[:sample_size].to(device) 
dataset = TensorDataset(embeddings, conditions, s_prime, label_idx)
train_size = int(percentage * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bs*2)


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


class Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(768, 200, batch_first=True)
        self.fc = nn.Linear(200, latent_dim)

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size, -1, 768)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.squeeze(0)

        return self.fc(h_n)


class NoiseEstimator(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim):
        super(NoiseEstimator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc3 = nn.Linear(256, input_dim)

    def forward(self, sigma):
        x = nn.functional.relu(self.fc1(sigma))
        noise = self.fc3(x)
        return noise


class Decoder(nn.Module):
    def __init__(self, latent_dim=100, length=50):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + 1536 + 768, 200)
        self.fc2 = nn.Linear(200, 400)
        self.fc3 = nn.Linear(400, 768)
        self.length = length

    def forward(self, z, condition, noise):
        z = z.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.length, 1)
        noise = noise.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.length, 1)
        z = torch.cat([z, condition, noise], dim=-1)
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))

        return torch.sigmoid(self.fc3(h2)).view(-1, 1, self.length, 768)


class CVQVAE(nn.Module):
    def __init__(self, latent_dim=100, num_embeddings=50, length=50):
        super(CVQVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, length)
        self.vq = VectorQuantizer(num_embeddings, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, condition):
        z_e = self.encoder(x)
        indices = self.vq(z_e)
        z_q, _ = self.vq.straight_through(z_e, indices)
        return self.decoder(z_q, condition)

###############################################################################
# Training loop
###############################################################################
lr = 1e-3
beta = 1
alpha = 200
num_epochs = 200
latent_dim = 10
input_dim = 768
hidden_dim = 200
reconstruction_loss = nn.MSELoss()
model = CVQVAE(latent_dim=latent_dim, num_embeddings=50, length=seqlen).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
flow_model = NoiseEstimator(latent_dim, input_dim, hidden_dim).to(device)
optimizer_spline = optim.Adam(flow_model.parameters(), lr=lr)

model.train()
start_time = time.time()
torch.cuda.empty_cache()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    recon_epoch_loss = 0.0
    commit_epoch_loss = 0.0
    epoch_start_time = time.time()

    train_estimate = []
    train_personality = []

    for batch_idx, (x, c, s_prime, label) in enumerate(train_loader):
        z_e = model.encoder(x)
        indices = model.vq(z_e)
        z_q, z_q_sg = model.vq.straight_through(z_e, indices)
        train_noise = flow_model(z_q_sg)
        outputs = model.decoder(z_q_sg, c, train_noise)

        rec_loss = reconstruction_loss(outputs, s_prime)
        commit_loss = F.mse_loss(z_q.detach(), z_e)
        quant_loss = F.mse_loss(z_e, z_q.detach())
        loss = rec_loss + alpha * commit_loss + beta * quant_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        recon_epoch_loss += rec_loss.item()
        commit_epoch_loss += commit_loss.item()
