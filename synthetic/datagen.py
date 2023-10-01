import csv
import torch
import random
import numpy as np
import torch.nn as nn
from scipy.stats import ortho_group
from scipy.stats import norm, expon, beta


noise_mean = 0
noise_std = 0.1
action_max = 1
num_iter = 30
num_users = 100
random_seed = 42
rng = np.random.default_rng(random_seed)

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_stable_parameters():
    while True:
        a = np.random.uniform(0, 1)  
        d = np.random.uniform(0, 1) 
        if abs(a) < 1 and abs(d) < 1:
            break

    b = np.random.choice([-1, 1]) * np.random.uniform(0, abs(1 - abs(a)))
    c = np.random.choice([-1, 1]) * np.random.uniform(0, abs(1 - abs(d)))
    
    return a, b, c, d
    
class tsSingleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(tsSingleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
    
class tsMultiMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(tsMultiMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class noiseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(noiseMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def logit(self, p):
        return np.log(p / (1 - p))

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
    
def generateUniformMat(Ncomp, condT):
    A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    while np.linalg.cond(A) > condT:
        A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    return A

def causalMask(n_nodes=5, seed=42, sparsity=0.6):
    np.random.seed(seed)
    mask = np.eye(n_nodes, dtype=int)
    num_nonzero = int(n_nodes * sparsity) - 1
    for i in range(n_nodes):
        choice = np.random.choice(n_nodes-1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        mask[i, choice] = 1
    return mask

# '''
## Case 1
num_action = 2
num_states = 3
num_latent = 1
dist = [0.1, 0.2, 0.3, 0.4]    
var_samples = random.choices([1, 2, 3, 4], dist, k=num_users)
L = [np.array((var)) for var in zip(var_samples)]

input_dim = num_states + num_action + num_latent
noise = noiseMLP(input_dim=num_latent+1, hidden_dim=2, output_dim=1)
model = tsMultiMLP(input_dim=input_dim, hidden_dim=10, output_dim=num_states)

with torch.no_grad():
    nn.init.xavier_uniform_(noise.fc1.weight)
    nn.init.zeros_(noise.fc1.bias)
    nn.init.xavier_uniform_(noise.fc2.weight)
    nn.init.zeros_(noise.fc2.bias)

with torch.no_grad():
    nn.init.xavier_uniform_(model.fc1.weight)
    nn.init.zeros_(model.fc1.bias)
    nn.init.xavier_uniform_(model.fc2.weight)
    nn.init.zeros_(model.fc2.bias)

for param in model.parameters():
    param.requires_grad = False

for param in noise.parameters():
    param.requires_grad = False

with open("./dataset/1d_discrete.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user", "l", "s", "a", "s'"]) 
    
    for user in range(num_users):
        trajectory = [ ]
        S = np.zeros((num_iter+1, num_states)) 
        A = np.random.rand(num_iter+1, num_action)
        S_0 = torch.randn(num_states).numpy()
        S[0] = S_0

        with torch.no_grad():
            noise_tensor = torch.FloatTensor(np.concatenate([L[user], np.array([noise_std])]))
            std = noise(noise_tensor).numpy()[0]

        for t in range(1, num_iter+1):
            input_data = np.concatenate([S[t-1], A[t-1], L[user]])
            input_tensor = torch.FloatTensor(input_data).reshape(1, -1)
            
            with torch.no_grad():
                S_t = np.tanh(model(input_tensor).numpy() + rng.normal(noise_mean, std, (1, num_states)))
            S_t = S_t.flatten().ravel()
            trajectory.append((L[user], S[t-1], A[t-1], S_t))
            
            S[t] = S_t

        for data_tuple in trajectory:
            data_write = (user,) + data_tuple
            writer.writerow(data_write)
# '''

'''
## Case 2
num_action = 2
num_states = 3
num_latent = 3

dist1 = [0.1, 0.9]            
dist2 = [0.2, 0.3, 0.5]        
dist3 = [0.1, 0.2, 0.3, 0.4]    
var1_samples = random.choices([1, 2], dist1, k=num_users)
var2_samples = random.choices([1, 2, 3], dist2, k=num_users)
var3_samples = random.choices([1, 2, 3, 4], dist3, k=num_users)
L = [np.array((var1, var2, var3)) for var1, var2, var3 in zip(var1_samples, var2_samples, var3_samples)]

input_dim = num_states + num_action + num_latent
noise = noiseMLP(input_dim=num_latent+1, hidden_dim=5, output_dim=1)
model = tsMultiMLP(input_dim=input_dim, hidden_dim=10, output_dim=num_states)
with torch.no_grad():
    nn.init.xavier_uniform_(noise.fc1.weight)
    nn.init.zeros_(noise.fc1.bias)
    nn.init.xavier_uniform_(noise.fc2.weight)
    nn.init.zeros_(noise.fc2.bias)
    
with torch.no_grad():
    nn.init.xavier_uniform_(model.fc1.weight)
    nn.init.zeros_(model.fc1.bias)
    nn.init.xavier_uniform_(model.fc2.weight)
    nn.init.zeros_(model.fc2.bias)

for param in model.parameters():
    param.requires_grad = False

with open("./dataset/md_discrete.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user", "l", "s", "a", "s'"])  
    
    for user in range(num_users):
        trajectory = [ ]
        S = np.zeros((num_iter+1, num_states)) 
        A = np.random.rand(num_iter+1, num_action)
        S_0 = torch.randn(num_states).numpy()
        S[0] = S_0
        
        with torch.no_grad():
            noise_tensor = torch.FloatTensor(np.concatenate([L[user], np.array([noise_std])]))
            std = noise(noise_tensor).numpy()[0]
        
        for t in range(1, num_iter+1):
            input_data = np.concatenate([S[t-1], A[t-1], L[user]])
            input_tensor = torch.FloatTensor(input_data).reshape(1, -1)

            with torch.no_grad():
                S_t = np.tanh(model(input_tensor).numpy() + rng.normal(noise_mean, std, (1, num_states)))
            S_t = S_t.flatten().ravel()
            trajectory.append((L[user], S[t-1], A[t-1], S_t))
            
            S[t] = S_t

        for data_tuple in trajectory:
            data_write = (user,) + data_tuple
            writer.writerow(data_write)
'''

'''
## Case 3
num_action = 2
num_states = 3
num_latent = 3

var1_samples = np.random.normal(0, 1, num_users)
var2_samples = np.random.uniform(0, 1, num_users)
var3_samples = np.random.exponential(1, num_users)
L = [np.array((var1, var2, var3)) for var1, var2, var3 in zip(var1_samples, var2_samples, var3_samples)]

input_dim = num_states + num_action + num_latent
noise = noiseMLP(input_dim=num_latent+1, hidden_dim=5, output_dim=1)
model = tsMultiMLP(input_dim=input_dim, hidden_dim=10, output_dim=num_states)
with torch.no_grad():
    nn.init.xavier_uniform_(noise.fc1.weight)
    nn.init.zeros_(noise.fc1.bias)
    nn.init.xavier_uniform_(noise.fc2.weight)
    nn.init.zeros_(noise.fc2.bias)
    
with torch.no_grad():
    nn.init.xavier_uniform_(model.fc1.weight)
    nn.init.zeros_(model.fc1.bias)
    nn.init.xavier_uniform_(model.fc2.weight)
    nn.init.zeros_(model.fc2.bias)

for param in model.parameters():
    param.requires_grad = False

with open("./dataset/md_continous.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user", "l", "s", "a", "s'"]) 
    
    for user in range(num_users):
        trajectory = [ ]
        S = np.zeros((num_iter+1, num_states)) 
        A = np.random.rand(num_iter+1, num_action)
        S_0 = torch.randn(num_states).numpy()
        S[0] = S_0
        
        with torch.no_grad():
            noise_tensor = torch.FloatTensor(np.concatenate([L[user], np.array([noise_std])]))
            std = noise(noise_tensor).numpy()[0]
        
        for t in range(1, num_iter+1):
            input_data = np.concatenate([S[t-1], A[t-1], L[user]])
            input_tensor = torch.FloatTensor(input_data).reshape(1, -1)

            with torch.no_grad():
                S_t = np.tanh(model(input_tensor).numpy() + rng.normal(noise_mean, std, (1, num_states)))
            S_t = S_t.flatten().ravel()
            trajectory.append((L[user], S[t-1], A[t-1], S_t))
            
            S[t] = S_t

        for data_tuple in trajectory:
            data_write = (user,) + data_tuple
            writer.writerow(data_write)
'''