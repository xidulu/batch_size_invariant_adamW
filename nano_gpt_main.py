import os
import time
import math
import pickle
from contextlib import nullcontext
import argparse

import numpy as np
import torch

from model import GPTConfig, GPT
from my_optim import SignSGD, AdaMoM, AdamBI
import torch.optim as optim


parser = argparse.ArgumentParser(description='NanoGPT Training')
parser.add_argument('--wd', default=0.01, type=float)
parser.add_argument('--LR', default=1e-4, type=float) # Base learning rate
parser.add_argument('--use_std_adam', action='store_true') # Use standard adam
parser.add_argument('--total_epoch', default=400, type=int)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--mbs', default=25, type=int) # Micro batch size
parser.add_argument('--bs', default=100, type=int) # Mini batch size
parser.add_argument('--gamma1', default=1e-2, type=float) # Base Gamma 1
parser.add_argument('--gamma2', default=1e-3, type=float) # Base Gamma 2
parser.add_argument('--KMAX', default=40, type=int) # Kappa max
args = parser.parse_args()


if args.seed:
    torch.manual_seed(args.seed)

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
bs = args.bs
mbs = args.mbs
total_epoch = args.total_epoch


block_size = 256 # context of up to 256 previous characters
len_data = 1003854
eval_iters = 500

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = args.LR # with baby networks can afford to go a bit higher
base_lr = learning_rate
weight_decay = args.wd


data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (mbs,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


device_type = 'cuda'
device = 'cuda'

ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float32)

iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=True, vocab_size=None, dropout=dropout) # start with model_args from command line

print("Initializing a new model from scratch")
# determine the vocab size we'll use for from-scratch training
if meta_vocab_size is None:
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device_type)


param_dict = {pn: p for pn, p in model.named_parameters()}
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
num_decay_params = sum(p.numel() for p in decay_params)
num_nodecay_params = sum(p.numel() for p in nodecay_params)


kappa = bs // mbs
KAPPA_MAX = args.KMAX
if args.use_std_adam:
    LR = base_lr * np.sqrt(kappa) / np.sqrt(KAPPA_MAX)
    gamma1, gamma2 = kappa * args.gamma1 / KAPPA_MAX, kappa * args.gamma2 / KAPPA_MAX
    beta1, beta2 = 1 - gamma1, 1 - gamma2
else:
    LR = base_lr * kappa / KAPPA_MAX
    gamma1, gamma2 = kappa * args.gamma1 / KAPPA_MAX, kappa * args.gamma2 / KAPPA_MAX

if args.use_std_adam:
    optimizer = optim.AdamW(optim_groups, lr=LR, betas=(beta1, beta2))
else: 
    optimizer = AdamBI(optim_groups, lr=LR, gammas=(gamma1, gamma2))

unoptimized_model = model
# model = torch.compile(model) # requires PyTorch 2.0
model = model

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model # unwrap DDP container if needed
running_mfu = -1.0

train_losses = []
test_losses = []


for epoch in range(total_epoch):
    # evaluate the loss on train/val sets and write checkpoints
        
    for i in range(len_data // (bs * block_size)):
        for micro_step in range(bs // mbs):
            with ctx:
                logits, loss = model(X, Y)
                if args.use_std_adam:
                    loss = loss / (bs // mbs) # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            loss.backward()
            if not args.use_std_adam:
                optimizer.accumlate(kappa)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    losses = estimate_loss()
    train_losses.append(losses['train'])
    test_losses.append(losses['val'])
    print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
prefix_string = 'NanoGPT'

prefix_string += f"lr_{base_lr}__mbs_{mbs}__bs_{bs}__kmax_{KAPPA_MAX}"


if args.use_std_adam:
    prefix_string += "adamSTD__"

if args.seed:
    prefix_string += f'seed_{args.seed}__'


np.save(f'./results/{prefix_string}_train.npy', np.array(train_losses))
np.save(f'./results/{prefix_string}_test.npy', np.array(test_losses))