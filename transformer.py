"""
Character-level Transformer language model from scratch.

This script contains:
- vocabulary building
- encoding / decoding
- batch creation
- causal self-attention
- single-head attention 
- multi-head attention
- feed-forward network
- Transformer block
- language model
- training loop
- text generation

"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# ============================================================
# Single attention head
# ============================================================

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v

        return out


# ============================================================
# Multi-head attention
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)

        return out


# ============================================================
# Feed-forward network
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Transformer block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()

        self.sa = MultiHeadAttention(n_embd, num_heads, block_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


# ============================================================
# Transformer language model
# ============================================================

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, num_heads, block_size, n_layers):
        super().__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(n_embd, num_heads, block_size)
                for _ in range(n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)

        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape

            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_new = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_new), dim=1)

        return idx


