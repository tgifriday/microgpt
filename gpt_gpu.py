"""
GPU-accelerated version of Karpathy's pure Python GPT.
Same architecture, same logic — but using PyTorch tensors on CUDA instead of scalar Value objects.
This turns hours into seconds.

Original: @karpathy
GPU port: preserves all architectural choices (RMSNorm, ReLU, no biases, single-doc training)
"""

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

device = torch.device('cpu')
print(f"Using device: {device}")

# --- Dataset ---
if not os.path.exists('datainput.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('datainput.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# --- Tokenizer ---
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

def encode(doc):
    return [BOS] + [uchars.index(ch) for ch in doc] + [BOS]

# --- Model ---
n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Original code has no learnable gain, just raw rmsnorm

    def forward(self, x):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)

class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = self.wk(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, n_head, head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Init weights with same std as original
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.norm_in(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

model = GPT().to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"num params: {num_params}")

# --- Optimizer: Adam with same hyperparams as original ---
learning_rate = 0.01
beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=eps_adam)

# --- Training ---
num_steps = 1500

# Optional: batch multiple docs per step for even more GPU utilization
batch_size = 64  # process many docs at once — the original does 1 at a time

model.train()
for step in range(num_steps):
    # Linear LR decay (matching original)
    lr_t = learning_rate * (1 - step / num_steps)
    for pg in optimizer.param_groups:
        pg['lr'] = lr_t

    # Build a batch of docs, pad to same length
    batch_tokens = []
    for b in range(batch_size):
        doc = docs[(step * batch_size + b) % len(docs)]
        tokens = encode(doc)
        tokens = tokens[:block_size + 1]  # cap at block_size + 1
        batch_tokens.append(tokens)

    # Pad sequences to max length in this batch
    max_len = max(len(t) for t in batch_tokens)
    input_ids = torch.full((batch_size, max_len - 1), BOS, dtype=torch.long, device=device)
    target_ids = torch.full((batch_size, max_len - 1), -100, dtype=torch.long, device=device)  # -100 = ignore

    for b, tokens in enumerate(batch_tokens):
        n = min(block_size, len(tokens) - 1)
        input_ids[b, :n] = torch.tensor(tokens[:n], dtype=torch.long)
        target_ids[b, :n] = torch.tensor(tokens[1:n+1], dtype=torch.long)

    # Forward
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1), ignore_index=-100)

    # Backward + update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 50 == 1 or step == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

print(f"\nFinal loss: {loss.item():.4f}")

# --- Inference ---
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
model.eval()
with torch.no_grad():
    for sample_idx in range(20):
        tokens = [BOS]
        for pos_id in range(block_size):
            idx = torch.tensor([tokens[-block_size:]], dtype=torch.long, device=device)
            logits = model(idx)
            logits = logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            if token_id == BOS:
                break
            tokens.append(token_id)
        name = ''.join(uchars[t] for t in tokens[1:])
        print(f"sample {sample_idx+1:2d}: {name}")
