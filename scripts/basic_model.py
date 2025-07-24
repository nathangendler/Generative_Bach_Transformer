import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

torch.manual_seed(1337)

# Load the musical tokens processed file
try:
    with open('./Transformer/music_tokens.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print("✅ Loaded music_tokens.txt")
except FileNotFoundError:
    try:
        with open('music_tokens.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        print("Loaded music tokens")
    except FileNotFoundError:
        print("Could not load music tokens")
        exit(1)

# split text into tokens
tokens = text.split()
print(f"Total musical tokens: {len(tokens)}")

# all unique tokens that appear
unique_tokens = sorted(list(set(tokens)))
vocab_size = len(unique_tokens)
print(f"Vocabulary size: {vocab_size} unique musical tokens")

# example tokens
print("\nExample musical tokens:")
for i, token in enumerate(unique_tokens[:10]):
    print(f"  {i}: {token}")

# create a mapping from tokens to integers
stoi = { token:i for i,token in enumerate(unique_tokens) }
itos = { i:token for i,token in enumerate(unique_tokens) }

# encoder and decoder
encode = lambda s: [stoi[token] for token in s.split()]
decode = lambda l: ' '.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Training data size: {len(train_data)} tokens")
print(f"Validation data size: {len(val_data)} tokens")

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# simple bigram model
class SimpleBigramMusicModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

print(f"Creating music model with vocabulary size: {vocab_size}")
model = SimpleBigramMusicModel(vocab_size)
m = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Simple model created with {total_params:,} parameters")

# PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"\nStarting simple Bach training on device: {device}")
print("Training progress:")

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training done. Generating music now:")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = m.generate(context, max_new_tokens=100)[0].tolist()
generated_music = decode(generated_tokens)

print("\n=== GENERATED SIMPLE BACH MUSIC ===")
print(generated_music)

# Save the generated music
try:
    with open('./Transformer/simple_generated_music.txt', 'w') as f:
        f.write(generated_music)
    print(f"\n generated music saved to './Transformer/simple_generated_music.txt'")
except:
    with open('simple_generated_music.txt', 'w') as f:
        f.write(generated_music)
    print(f"\n generated music saved to 'simple_generated_music.txt'")

try:
    torch.save(model.state_dict(), './Transformer/simple_music_model.pth')
    print(f" model saved to './Transformer/simple_music_model.pth'")
except:
    torch.save(model.state_dict(), 'simple_music_model.pth')
    print(f" model saved to 'simple_music_model.pth'")
