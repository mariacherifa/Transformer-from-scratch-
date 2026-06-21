
import torch 
from transformer import Head, MultiHeadAttention, TransformerBlock, TransformerLanguageModel
from tokenizer import build_vocab, encode, decode
from data import create_data, get_batch
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
