import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics
from torch.autograd import Variable
import argparse
import csv

# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SYN')
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--dev", help="device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--epochs", type=int, help="number of epochs for ae", default=5000)
parser.add_argument("--lr", type=float, help="learning rate", default=1e-2)
parser.add_argument("--memlen", type=int, help="size of memory", default=16)
args = parser.parse_args()

# Synthetic data paths
if args.dataset == 'SYN':
    nfile = '/content/drive/MyDrive/FTP/data/syn.txt'
    lfile = '/content/drive/MyDrive/FTP/data/synlabel.txt'

# Device setup
device = torch.device(args.dev)

# Define the MemStream class
class MemStream(nn.Module):
    def __init__(self, in_dim, params):
        super(MemStream, self).__init__()
        self.params = params
        self.in_dim = in_dim
        self.out_dim = 1
        self.memory_len = params['memory_len']
        self.max_thres = torch.tensor(params['beta']).to(device)
        self.memory = torch.randn(self.memory_len, self.out_dim).to(device)
        self.mem_data = torch.randn(self.memory_len, self.in_dim).to(device)
        self.mem_idx = torch.arange(self.memory_len).to(device)
        self.memory.requires_grad = False
        self.mem_data.requires_grad = False
        self.mem_idx.requires_grad = False
        self.batch_size = params['memory_len']
        self.num_mem_update = 0
        self.encoder = nn.Identity().to(device)
        self.decoder = nn.Identity().to(device)
        self.clock = 0
        self.last_update = -1
        self.updates = []
        self.loss_fn = nn.MSELoss()
        self.count = 0

    def train_autoencoder(self, data, epochs):
        self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
        # Your training logic here (placeholder)

    def update_memory(self, output_loss, encoder_output, data):
        if output_loss <= self.max_thres:
            least_used_pos = torch.argmin(self.mem_idx)
            self.memory[least_used_pos] = encoder_output
            self.mem_data[least_used_pos] = data
            self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
            self.mem_idx[least_used_pos] = self.count
            self.count += 1
            self.num_mem_update += 1
            return 1
        return 0

    def initialize_memory(self, x):
        mean, std = self.mem_data.mean(0), self.mem_data.std(0)
        new = (x - mean) / std
        new[:, std == 0] = 0
        self.memory = self.encoder(new)
        self.memory.requires_grad = False
        self.mem_data = x

    def forward(self, x):
        new = (x - self.mean) / self.std
        new[:, self.std == 0] = 0
        encoder_output = self.encoder(new)
        loss_values = torch.norm(self.memory - encoder_output, dim=1, p=1).min()
        self.updates.append(self.update_memory(loss_values, encoder_output, x))
        return loss_values

# Load synthetic data
numeric = torch.FloatTensor(np.loadtxt(nfile, delimiter=',')).reshape(-1, 1).to(device)
labels = np.loadtxt(lfile, delimiter=',')

# Set random seed
torch.manual_seed(args.seed)

# Parameters
N = args.memlen
params = {
    'beta': args.beta,
    'memory_len': N,
    'batch_size': 1,
    'lr': args.lr
}

# Initialize model
model = MemStream(numeric[0].shape[0], params).to(device)
model.max_thres = model.max_thres.float()

# Training and initialization
batch_size = params['batch_size']
print(args.dataset, args.beta, args.memlen, args.lr, args.epochs)
data_loader = DataLoader(numeric, batch_size=batch_size)
init_data = numeric[labels == 0][:N].to(device)
model.mem_data = init_data

# Training phase
torch.set_grad_enabled(True)
model.train_autoencoder(Variable(init_data).to(device), epochs=args.epochs)
torch.set_grad_enabled(False)
model.initialize_memory(Variable(init_data[:N]))

# Evaluation and metrics
err = []
l0 = 0
l1 = 0

for data, label in zip(data_loader, labels):
    output = model(data.to(device))
    err.append(output)

    mean, std = model.mem_data.mean(0), model.mem_data.std(0)
    new = (data.to(device) - mean) / std
    new[:, std == 0] = 0

    if ((new > 2).any() or (new < -2).any()) and label == 0:
        l0 += 1
    if ((new <= 2).all() and (new >= -2).all()) and label == 1:
        l1 += 1

scores = np.array([i.cpu().detach().numpy() for i in err])
auc = metrics.roc_auc_score(labels, scores)
pr = metrics.average_precision_score(labels, scores)

print("MemStream")
print("ROC-AUC:", auc)
print("AUC-PR:", pr)
print()

# Evaluate false positives (l0) and false negatives (l1)
print("False Positives (l0):", l0)
print("False Negatives (l1):", l1)