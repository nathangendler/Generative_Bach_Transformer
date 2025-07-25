import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Training parameters
batch_size = 64 
block_size = 256 
epochs = 50  
eval_interval = 1  # Changed to 1 to evaluate after every epoch
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 
n_head = 12 
n_layer = 8 
dropout = 0.2

if torch.cuda.is_available():
    print(f"device: cuda")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

try:
    with open('../tokens/music_tokens.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print("Loaded music tokens")
except FileNotFoundError:
    try:
        with open('tokens/music_tokens.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        print("Loaded music tokens from root directory")
    except FileNotFoundError:
        try:
            with open('music_tokens.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            print("Loaded music tokens from current directory")
        except FileNotFoundError:
            print("Could not find music tokens file")
            print("Checked: ../tokens/music_tokens.txt, tokens/music_tokens.txt, and music_tokens.txt")
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

# Create proper datasets and dataloaders
def create_sequences(data, block_size):
    """Create input-target sequence pairs"""
    sequences = []
    targets = []
    for i in range(len(data) - block_size):
        sequences.append(data[i:i+block_size])
        targets.append(data[i+1:i+block_size+1])
    return torch.stack(sequences), torch.stack(targets)

# Create training and validation datasets
train_sequences, train_targets = create_sequences(train_data, block_size)
val_sequences, val_targets = create_sequences(val_data, block_size)

train_dataset = TensorDataset(train_sequences, train_targets)
val_dataset = TensorDataset(val_sequences, val_targets)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Training sequences: {len(train_dataset):,}")
print(f"Validation sequences: {len(val_dataset):,}")

@torch.no_grad()
def validate_model(model, val_dataloader, device):
    """Validation function"""
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    
    for X, Y in val_dataloader:
        X, Y = X.to(device), Y.to(device)
        logits, loss = model(X, Y)
        total_val_loss += loss.item()
        num_val_batches += 1
        
        # Limit validation batches for speed
        if num_val_batches >= eval_iters:
            break
    
    model.train()
    return total_val_loss / num_val_batches

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size 
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * self.head_size**-0.5 
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
        self.ln_f = nn.LayerNorm(n_embd)
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
        x = tok_emb + pos_emb
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

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

print(f"Model parameters: {total_params/1e6:.1f}M")

best_val_loss = float('inf')
patience_counter = 0
patience = 5 
train_loss_history = []
val_loss_history = []
print("\nStarting training")

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0.0
    num_train_batches = 0
    
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for batch_idx, (X, Y) in enumerate(pbar):
        X, Y = X.to(device), Y.to(device)
        
        logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_train_loss += loss.item()
        num_train_batches += 1
        
        pbar.set_postfix(loss=loss.item())
    
    scheduler.step()
    avg_train_loss = epoch_train_loss / num_train_batches
    train_loss_history.append(avg_train_loss)
    
    val_loss = validate_model(model, val_dataloader, device)
    val_loss_history.append(val_loss)
    current_lr = scheduler.get_last_lr()[0]
    
    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
    print(f"Learning Rate: {current_lr:.2e}")
    
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e6
        memory_cached = torch.cuda.memory_reserved() / 1e6
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        try:
            torch.save(model.state_dict(), '../models/bach_model.pth')
        except:
            try:
                torch.save(model.state_dict(), 'models/bach_model.pth')
            except:
                torch.save(model.state_dict(), 'bach_model.pth')
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter}/{patience} epochs")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss achieved: {best_val_loss:.6f}")
            try:
                torch.save(model.state_dict(), '../models/bach_model_final.pth')
            except:
                try:
                    torch.save(model.state_dict(), 'models/bach_model_final.pth')
                except:
                    torch.save(model.state_dict(), 'bach_model_final.pth')
            break
        
    # Clear GPU cache periodically
    if torch.cuda.is_available() and epoch % 10 == 0:
        torch.cuda.empty_cache()

print(f"\nTraining Summary:")
print(f"Total epochs completed: {len(train_loss_history)}")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Final training loss: {train_loss_history[-1]:.6f}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load the best model for generation
try:
    model.load_state_dict(torch.load('../models/bach_model.pth'))
except:
    try:
        model.load_state_dict(torch.load('models/bach_model.pth'))
    except:
        model.load_state_dict(torch.load('bach_model.pth'))

model.eval()

print(f"\nGenerating music sequence")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = model.generate(context, max_new_tokens=200)[0].tolist()
generated_music = decode(generated_tokens)

try:
    with open('/generated/generated_music.txt', 'w') as f:
        f.write(generated_music)
except:
    try:
        with open('generated/generated_music.txt', 'w') as f:
            f.write(generated_music)
    except:
        with open('generated_music.txt', 'w') as f:
            f.write(generated_music)

try:
    torch.save(model.state_dict(), '/models/bach_gpt_model.pth')
except:
    try:
        torch.save(model.state_dict(), 'models/bach_gpt_model.pth')
    except:
        torch.save(model.state_dict(), 'bach_gpt_model.pth')

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nTraining completed successfully!")