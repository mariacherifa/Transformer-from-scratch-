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
# Configuration
# ============================================================

batch_size = 32
block_size = 8
max_iters = 1000
eval_interval = 100
eval_iters = 50
learning_rate = 1e-3

n_embd = 64
num_heads = 4
n_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Vocabulary and tokenization
# ============================================================

def build_vocab(text):
    vocabulary = sorted(set(text))
    vocab_size = len(vocabulary)

    itos = {}
    stoi = {}

    for i, ch in enumerate(vocabulary):
        itos[i] = ch
        stoi[ch] = i

    return vocabulary, vocab_size, itos, stoi


def encode(text, stoi):
    result = []

    for ch in text:
        result.append(stoi[ch])

    return result


def decode(ids, itos):
    result = ""

    for i in ids:
        result += itos[int(i)]

    return result


# ============================================================
# Data creation and batching
# ============================================================

def create_data(text, stoi):
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return data, train_data, val_data


def get_batch(data, batch_size, block_size, device):
    if len(data) <= block_size:
        raise ValueError("data is too short for the chosen block_size")
    
    idx = torch.randint(0, len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in idx])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in idx])

    x = x.to(device)
    y = y.to(device)

    return x, y


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


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    result = {}

    model.eval()

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []

        for _ in range(eval_iters):
            xb, yb = get_batch(data, batch_size, block_size, device)
            logits, loss = model(xb, yb)
            losses.append(loss.item())

        result[split] = sum(losses) / len(losses)

    model.train()

    return result


# ============================================================
# Training
# ============================================================

def train_model(model, train_data, val_data):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        if step % eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                batch_size,
                block_size,
                eval_iters,
                device,
            )

            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch(train_data, batch_size, block_size, device)

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ============================================================
# Main script
# ============================================================

if __name__ == "__main__":
    text = (
        "Hello my name is Maria Cherifa. "
        "I have been working on language models. "
        "This is a small character-level Transformer from scratch. "
    ) * 100

    vocabulary, vocab_size, itos, stoi = build_vocab(text)
    data, train_data, val_data = create_data(text, stoi)

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        num_heads=num_heads,
        block_size=block_size,
        n_layers=n_layers,
    ).to(device)

    train_model(model, train_data, val_data)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_ids = model.generate(context, max_new_tokens=300)

    generated_text = decode(generated_ids[0].tolist(), itos)

    print("\nGenerated text:")
    print(generated_text)
