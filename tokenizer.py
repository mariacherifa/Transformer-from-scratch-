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
