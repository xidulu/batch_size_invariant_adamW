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

import os
import argparse
from my_optim import SignSGD, AdaMoM, AdamBI
from models import *
from autoaugment import CIFAR10Policy


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

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
parser.add_argument('--use_std_adam', action='store_true') # Use standard adam
parser.add_argument('--total_epoch', default=400, type=int)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--use_data_aug', action='store_true')
parser.add_argument('--mbs', default=25, type=int) # Micro batch size
parser.add_argument('--bs', default=100, type=int) # Mini batch size
parser.add_argument('--gamma1', default=1e-2, type=float) # Base Gamma 1
parser.add_argument('--gamma2', default=1e-3, type=float) # Base Gamma 2
parser.add_argument('--KMAX', default=40, type=int) # Kappa max
parser.add_argument('--use_grad_accum', action='store_true') # Use gradient accumlation
args = parser.parse_args()


total_epoch = args.total_epoch
seed = args.seed
use_data_aug = args.use_data_aug
bs = args.bs
mbs = args.mbs
wd = args.wd
kappa = bs // mbs
KAPPA_MAX = args.KMAX
base_lr = args.LR
if args.use_std_adam:
    LR = base_lr * np.sqrt(kappa) / np.sqrt(KAPPA_MAX)
    gamma1, gamma2 = kappa * args.gamma1 / KAPPA_MAX, kappa * args.gamma2 / KAPPA_MAX
    beta1, beta2 = 1 - gamma1, 1 - gamma2
else:
    LR = base_lr * kappa / KAPPA_MAX
    gamma1, gamma2 = kappa * args.gamma1 / KAPPA_MAX, kappa * args.gamma2 / KAPPA_MAX


if seed:
    torch.manual_seed(seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
if use_data_aug:
    print('==> Use data augmentation!')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        CIFAR10Policy(),
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

# Model
print('==> Building model..')

net = ViT(patch=8)

net = net.to(device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# criterion = nn.CrossEntropyLoss()
criterion = LabelSmoothingCrossEntropyLoss(10, 0.1)
if args.use_std_adam:
    optimizer = optim.AdamW(net.parameters(), lr=LR, betas=(beta1, beta2), weight_decay=0.01)
else: 
    optimizer = AdamBI(net.parameters(), lr=LR, gammas=(gamma1, gamma2), weight_decay=0.01)

def train_micro_batch(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_losses = []
    correct = 0
    total = 0
    for batch_idx, (_inputs, _targets) in enumerate(trainloader):
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
    return correct / total, np.mean(train_losses)


def train_normal(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_losses = []
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        train_losses.append(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        optimizer.step()
    return correct / total, np.mean(train_losses)


def train_big_batch(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_losses = []
    correct = 0
    total = 0
    MBS=100
    for batch_idx, (_inputs, _targets) in enumerate(trainloader):
        optimizer.zero_grad()
        for inputs, targets in zip(torch.split(_inputs, MBS), torch.split(_targets, MBS)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets) / (kappa)
            loss.backward()
            train_losses.append(loss.item() * kappa)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # print(np.mean(train_losses)) 
        optimizer.step()  
    return correct / total, np.mean(train_losses)



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
for epoch in range(total_epoch):
    if args.use_std_adam:
        if args.use_grad_accum:
            train_acc, train_loss = train_big_batch(epoch)
        else:
            train_acc, train_loss = train_normal(epoch)
    else:    
        train_acc, train_loss = train_micro_batch(epoch)
    test_acc, test_loss = test(epoch)
    train_records.append((train_acc, train_loss))
    test_records.append((test_acc, test_loss))

prefix_string = 'ViT'

prefix_string += f"lr_{base_lr}__mbs_{mbs}__bs_{bs}__kmax_{KAPPA_MAX}"


if use_data_aug:
    prefix_string += 'dataAug__'

if args.use_std_adam:
    prefix_string += "adamSTD__"

if seed:
    prefix_string += f'seed_{seed}__'


np.save(f'./results/{prefix_string}_train.npy', np.array(train_records))
np.save(f'./results/{prefix_string}_test.npy', np.array(test_records))
