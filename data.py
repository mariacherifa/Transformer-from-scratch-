# ============================================================
# Data creation and batching
# ============================================================
import torch
from tokenizer import build_vocab, encode, decode 
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

