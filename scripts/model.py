import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64 
block_size = 256 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 
n_head = 12 
n_layer = 8 
dropout = 0.2
# ------------

if torch.cuda.is_available():
    print(f"device: cuda")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# Load the musical tokens from your processed file
try:
    with open('../tokens/music_tokens.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    try:
        with open('music_tokens.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        print("Loaded music tokens")
    except FileNotFoundError:
        print("Could not find music tokens file")
        exit(1)

tokens = text.split()
print(f"Total musical tokens: {len(tokens)}")

unique_tokens = sorted(list(set(tokens)))
vocab_size = len(unique_tokens)


# create a mapping from tokens to integers
stoi = { token:i for i,token in enumerate(unique_tokens) }
itos = { i:token for i,token in enumerate(unique_tokens) }

encode = lambda s: [stoi[token] for token in s.split()]
decode = lambda l: ' '.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

print(f"Training data size: {len(train_data)} tokens")
print(f"Validation data size: {len(val_data)} tokens")

def get_batch(split):
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

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)  
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * head_size**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.sa(self.ln1(x)))
        x = x + self.dropout2(self.ffwd(self.ln2(x)))
        return x

class MusicalGPTModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb  # Add token and positional embeddings
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = MusicalGPTModel()
m = model.to(device)

total_params = sum(p.numel() for p in m.parameters())
print(f"Model created with {total_params/1e6:.1f}M parameters")


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

print("Training progress:")

best_val_loss = float('inf')
patience_counter = 0
patience = 5  

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {current_lr:.2e}")
        
        # GPU memory monitoring
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e6
            memory_cached = torch.cuda.memory_reserved() / 1e6
            print(f"         GPU memory: {memory_used:.1f}MB used, {memory_cached:.1f}MB cached")
        
        # Early stopping logic
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0
            # Save best model
            try:
                torch.save(model.state_dict(), '../models/bach_model.pth')
            except:
                torch.save(model.state_dict(), 'bach_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at iteration {iter} (no improvement for {patience} evaluations)")
                break

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Gradient clipping for stable training
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()  # Update learning rate
    
    # Clear GPU cache periodically
    if torch.cuda.is_available() and iter % 100 == 0:
        torch.cuda.empty_cache()

print("\nTraining complete! Generating new Bach-style compositions...")

# Clear GPU memory before generation
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load best model for generation
try:
    model.load_state_dict(torch.load('../models/bach_model.pth'))
    print("Loaded best model for generation")
except:
    try:
        model.load_state_dict(torch.load('bach_model.pth'))
        print("Loaded best model for generation")
    except:
        print("Using final model for generation")

model.eval()  # Set to evaluation mode

# Generate different length pieces
lengths = [50, 100, 200]
for i, length in enumerate(lengths, 1):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=length)[0].tolist()
    generated_music = decode(generated_tokens)
    
    print(f"\n=== GENERATED COMPOSITION #{i} ({length} tokens) ===")
    
    # Show first and last few tokens for readability
    tokens_list = generated_music.split()
    if len(tokens_list) <= 20:
        print(generated_music)
    else:
        first_10 = ' '.join(tokens_list[:10])
        last_10 = ' '.join(tokens_list[-10:])
        print(f"Start: {first_10}")
        print(f"  ... ({len(tokens_list)-20} more tokens) ...")
        print(f"End:   {last_10}")
    
    # Save each composition
    filename = f'../generated/generated_bach_{length}_tokens.txt'
    try:
        with open(filename, 'w') as f:
            f.write(generated_music)
        print(f"Saved to {filename}")
    except:
        filename = f'generated_bach_{length}_tokens.txt'
        with open(filename, 'w') as f:
            f.write(generated_music)
        print(f"Saved to {filename}")

# Save the trained model
try:
    torch.save(model.state_dict(), '../models/bach_gpt_model.pth')
    print(f"\nModel saved to '../models/bach_gpt_model.pth'")
except:
    torch.save(model.state_dict(), 'bach_gpt_model.pth')
    print(f"\nModel saved to 'bach_gpt_model.pth'")

print("\nYour Advanced Bach AI Composer is ready!")
print("The model has learned Bach's musical patterns and can generate new compositions.")

# Final GPU memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated() / 1e6
    print(f"Final GPU memory usage: {final_memory:.1f} MB")

print("Run this script again to generate more unique pieces!")

# Optional: Generate a really long piece
print(f"\nGenerating an extended composition (500 tokens)...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
long_piece = model.generate(context, max_new_tokens=500)[0].tolist()
long_music = decode(long_piece)

try:
    with open('../generated/bach_long_composition.txt', 'w') as f:
        f.write(long_music)
    print(f"Extended composition saved to '../generated/bach_long_composition.txt'")
except:
    with open('bach_long_composition.txt', 'w') as f:
        f.write(long_music)
    print(f"Extended composition saved to 'bach_long_composition.txt'")