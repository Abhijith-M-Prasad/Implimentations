import torch
import torch.nn as nn
import torch.nn.functional as F

# defalut hyperparameters
BATCH_SIZE = 64
CONTEXT_LENGTH = 256
EMBEDDING_SIZE = 384
HEAD_NO = 6
LAYER_NO = 6
HEAD_SIZE = EMBEDDING_SIZE//HEAD_NO
DROPOUT = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 5000
EVAL_ITR = 500
EVAL_INTERVAL = 200
LR = 3e-4
optim = torch.optim.AdamW



class Head(nn.Module):
    """ a single head of self attention. """

    def __init__(self):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
        self.query = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
        self.value = nn.Linear(EMBEDDING_SIZE, HEAD_SIZE, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CONTEXT_LENGTH,CONTEXT_LENGTH)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)#BxTxd
        k = self.key(x)#BxTxd
        w = (q @ k.mT)#BxTxd @ BxdxT => BxTxT
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))/(HEAD_SIZE**0.5)
        w = F.softmax(w, dim=-1)#BxTxT
        w = self.dropout(w)#BxTxT
        v = self.value(x)#BxTxd
        out  = w @ v #BxTxd
        return out
    
class Multi_Head_Attention(nn.Module):
    """ multiple attention head """
    
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(HEAD_NO)])
        self.proj = nn.Linear(HEAD_SIZE*HEAD_NO, EMBEDDING_SIZE)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

    
class MLP(nn.Module):
    """ a simple feed forward network """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE), 
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ a transformer decoder block """

    def __init__(self):
        super().__init__()
        self.heads = Multi_Head_Attention()
        self.ffn = MLP()
        self.ln1 = nn.LayerNorm(EMBEDDING_SIZE)
        self.ln2 = nn.LayerNorm(EMBEDDING_SIZE)

    def forward(self, x):
        x = x + self.heads(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):

    """ the full transformer model """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)
        self.pos_embedding = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_SIZE)
        self.blocks = nn.Sequential(*[Block() for _ in range(LAYER_NO)])
        self.ln_f = nn.LayerNorm(EMBEDDING_SIZE)
        self.lm_head = nn.Linear(EMBEDDING_SIZE, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        p = self.pos_embedding(torch.arange(T, device=idx.device)) # B,T,Embed
        t = self.token_embedding(idx) # B,T,Embed
        out = self.blocks(p + t) # B,T,Embed
        logits = self.lm_head(out) # B,T,vocab_size

        if targets is None:
            loss = None
        
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_token):
        for _ in range(max_token):
            # clipping the input to get context length from last 
            idx_cond = idx[:, -CONTEXT_LENGTH:]
            # forward propagate
            logits, _ = self(idx_cond)
            # probablities from logits
            logits = logits[:, -1, :]
            prob = F.softmax(logits, dim=-1)
            # sample from multinomial distribution
            idx_nxt = torch.multinomial(prob, num_samples=1)
            # append to previous context
            idx = torch.cat([idx, idx_nxt], dim=-1)# B,T+1
        return idx
