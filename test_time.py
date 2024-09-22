'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import time

import os
import argparse
from my_optim import SignSGD, AdaMoM, AdamBI
from models import *


def get_cifar_subset_loader(trainset_full, subset_size, bs):
    np.random.seed(args.seed)
    class_indices = [[] for _ in range(10)]  # CIFAR10 has 10 classes
    for idx, (_, label) in enumerate(trainset_full):
        class_indices[label].append(idx)
    num_samples_per_class = subset_size // 10  # Change this as needed
    balanced_subset_indices = []
    for i in range(10):
        balanced_subset_indices.extend(np.random.choice(class_indices[i], num_samples_per_class, replace=False))
    np.random.shuffle(balanced_subset_indices)
    trainset_subset = Subset(trainset_full, balanced_subset_indices)
    trainloader_subset = torch.utils.data.DataLoader(trainset_subset, batch_size=bs,
                                                    shuffle=True, num_workers=2)
    return trainloader_subset



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--wd', default=None, type=float)
parser.add_argument('--LR', default=1e-4, type=float) # Base learning rate
parser.add_argument('--layernorm', action='store_true') # Use LayerNorm model
parser.add_argument('--use_bi_adam', action='store_true') # Use Batchsize invariant adam
parser.add_argument('--total_epoch', default=400, type=int)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--use_data_aug', action='store_true')
parser.add_argument('--mbs', default=25, type=int) # Micro batch size
parser.add_argument('--bs', default=100, type=int) # Mini batch size
parser.add_argument('--gamma1', default=1e-1, type=float) # Base Gamma 1
parser.add_argument('--gamma2', default=1e-3, type=float) # Base Gamma 2
args = parser.parse_args()


total_epoch = args.total_epoch
seed = args.seed
use_data_aug = args.use_data_aug
bs = args.bs
mbs = args.mbs
wd = args.wd
kappa = bs // mbs
KAPPA_MAX = 40
base_lr = args.LR
LR = base_lr * kappa / KAPPA_MAX
gamma1, gamma2 = kappa * args.gamma1 / KAPPA_MAX, kappa * args.gamma2 / KAPPA_MAX


if seed:
    torch.manual_seed(seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if use_data_aug:
    print('==> Use data augmentation!')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    print('==> No data augmentation!')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=bs, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

if args.layernorm:
    net = ResNet18GnormLN()
else:
    net = ResNet18()

net = net.to(device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = AdamBI(net.parameters(), lr=LR, gammas=(gamma1, gamma2))

def train_micro_batch(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_losses = []
    correct = 0
    total = 0
    all_times = []
    for batch_idx, (_inputs, _targets) in enumerate(trainloader):
        start = time.time()
        for inputs, targets in zip(torch.split(_inputs, mbs), torch.split(_targets, mbs)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.accumlate(kappa)
            train_losses.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        optimizer.step()
        end = time.time()
        all_times.append(end - start)
    return np.array(all_times)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total, test_loss / (batch_idx+1)

train_records = []
test_records = []
times = train_micro_batch(0)
np.save(f'./results/time_{mbs}_{bs}.npy', times)




# np.save(f'./results/{prefix_string}_train.npy', np.array(train_records))
# np.save(f'./results/{prefix_string}_test.npy', np.array(test_records))
