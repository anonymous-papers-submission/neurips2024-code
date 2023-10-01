import os
import ast
import csv
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split


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
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, sigma):
        x = nn.functional.relu(self.fc1(sigma))
        noise = self.fc2(x)
        return noise

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

def convert_to_float_or_list(item):
    try:
        return float(item)
    except ValueError:
        try:
            return list(map(float, ast.literal_eval(item)))
        except (ValueError, SyntaxError):
            return item

###############################################################################
# DataLoader
###############################################################################
input_dim = 3
condition_dim = 5
latent_dim = 3
num_embeddings = 30

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_type = 'md_discrete'
ckp_path = "./checkpoint/"+ data_type + ".pth"
csv_file_relative_path = os.path.join("..", "synthetic/dataset", data_type + ".csv")
pickle_file_relative_path = os.path.join("..", "synthetic/dataset", data_type + "_vqvae_list.pkl")

current_directory = os.getcwd()
csv_file_path = os.path.abspath(os.path.join(current_directory, csv_file_relative_path))

extracted_data = [ ]
with open(csv_file_path, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        data_tuple = tuple([convert_to_float_or_list(x) for x in row])
        extracted_data.append(data_tuple)

idx = torch.tensor([row[0] for row in extracted_data])
latent = torch.tensor(np.array([np.fromstring(row[1][1:-1], sep=' ') for row in extracted_data]))
state = torch.tensor(np.array([np.fromstring(row[2][1:-1], sep=' ') for row in extracted_data]))
action = torch.tensor(np.array([np.fromstring(row[3][1:-1], sep=' ') for row in extracted_data]))
next_state = torch.tensor(np.array([np.fromstring(row[4][1:-1], sep=' ') for row in extracted_data]))
labels = torch.cat([state, action], dim=1).to(device).float()
idx = idx.unsqueeze(1).to(device).float()
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
next_state = next_state.to(device).float()
if latent_dim == 1:
    latent = latent.unsqueeze(1).to(device)
else:
    latent = latent.to(device).float()

batch_size = 32
percentage = 0.8
dataset = TensorDataset(idx, states, next_state, labels, latent)
train_size = int(percentage * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size*2)


###############################################################################
# Training loop
###############################################################################
beta = 1
alpha = 20
num_layers = 2
hidden_dim = 32
num_epochs = 200
learning_rate = 1e-4
model = ConditionalVQVAE(input_dim, condition_dim, hidden_dim, latent_dim, num_embeddings, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
reconstruction_loss = nn.MSELoss()
flow_model = NoiseEstimator(latent_dim, input_dim, hidden_dim).to(device)
optimizer_spline = optim.Adam(flow_model.parameters(), lr=learning_rate)

def TrainCCA(estimateds, latents, train_or_not):
    estimateds = torch.cat(estimateds, dim=0).cpu().detach().numpy()
    latents = np.concatenate(latents, axis=0)
    print(f"Rank of X on the training: {np.linalg.matrix_rank(latents)},", f"Rank of Y on the training: {np.linalg.matrix_rank(estimateds)}")
    latents = scaler_x.fit_transform(latents)
    estimateds = scaler_y.fit_transform(estimateds)
    cca.fit(latents, estimateds)
    if train_or_not == True:
        x_train_c, y_train_c = cca.transform(latents, estimateds)
        correlations = np.corrcoef(x_train_c.T, y_train_c.T)[:x_train_c.shape[1], x_train_c.shape[1]:]
        print("Canonical correlations on training set:", np.diag(correlations))

def TestCCA(estimateds, latents):
    print(f"Rank of X on the testing: {np.linalg.matrix_rank(latents)},", f"Rank of Y on the testing: {np.linalg.matrix_rank(estimateds)}")
    estimateds = scaler_y.fit_transform(estimateds)
    latents = scaler_x.fit_transform(latents)
    x_val_c, y_val_c = cca.transform(latents, estimateds)
    canonical_correlations = np.array([np.corrcoef(x_val_c[:, i], y_val_c[:, i])[0, 1] for i in range(x_val_c.shape[1])])
    print("Canonical correlations on the testing set:", canonical_correlations)

    return x_val_c, y_val_c, canonical_correlations

model.train()
flow_model.train()
torch.cuda.empty_cache()
scaler_x = StandardScaler()
scaler_y = StandardScaler()
# Training process
pcc_list = []
start_time = time.time()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    recon_epoch_loss = 0.0
    commit_epoch_loss = 0.0
    train_latents = [ ]
    train_estimateds = [ ]
    cca = CCA(n_components=3) 
    epoch_start_time = time.time()

    for idx, states, next_state, labels, latent in train_loader:
        z_e = model.encoder(states)
        indices = model.vq(z_e)
        z_q, z_q_sg = model.vq.straight_through(z_e, indices)
        train_noise = flow_model(z_q_sg)
        outputs = model.decoder(z_q_sg, labels, train_noise)

        rec_loss = reconstruction_loss(outputs, next_state)
        commit_loss = F.mse_loss(z_q.detach(), z_e)
        quant_loss = F.mse_loss(z_e, z_q.detach())
        loss = rec_loss + alpha * commit_loss + beta * quant_loss

        optimizer.zero_grad()
        optimizer_spline.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_spline.step()

        epoch_loss += loss.item()
        recon_epoch_loss += rec_loss.item()
        commit_epoch_loss += commit_loss.item()

        if latent_dim != 1:
            train_latents.append(latent.cpu().squeeze().numpy())
            train_estimateds.append(z_q)

    TrainCCA(train_estimateds, train_latents, False)

    model.eval()
    flow_model.eval()
    val_losses = 0
    val_latents = [ ]
    val_estimateds = [ ]
    with torch.no_grad():
        for _, val_states, val_next_state, val_labels, val_latent in val_loader:
            val_z_e = model.encoder(val_states)
            val_indices = model.vq(val_z_e)
            val_z_q, val_z_q_sg = model.vq.straight_through(val_z_e, val_indices)
            val_noise = flow_model(val_z_q_sg)
            val_outputs = model.decoder(val_z_q_sg, val_labels, val_noise)

            val_rec_loss = reconstruction_loss(val_outputs, val_next_state)
            val_commit_loss = F.mse_loss(val_z_q.detach(), val_z_e)
            val_quant_loss = F.mse_loss(val_z_e, val_z_q.detach())
            val_loss = val_rec_loss + alpha * val_commit_loss + beta * val_quant_loss

            val_losses += val_loss.item()
            val_latents.append(val_latent.cpu().squeeze().numpy())
            val_estimateds.append(val_z_q.cpu().squeeze().numpy())

    val_latents = np.concatenate(val_latents)
    val_estimateds = np.concatenate(val_estimateds)
    _, _, canonical_correlations = TestCCA(val_estimateds, val_latents)
    pcc_list.append(canonical_correlations)
    flow_model.train()
    model.train()
