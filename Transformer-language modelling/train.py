import torch
import pkg.transformer as tr

def path()->str:
    return "D:\Github\Dataset\TinyShakesphere.txt"

with open(path(), 'r', encoding='utf-8') as f:
    text = f.read()

vocab = list(set(text))
vocab_size = len(vocab)
def encode(text):
    chartoi = {char:i for i, char in enumerate(vocab)}
    return [chartoi[char] for char in text]
def decode(idx_lst):
    itochar = {i:char for i, char in enumerate(vocab)}
    return ''.join([itochar[i] for i in idx_lst])

data = torch.tensor(encode(text), dtype=torch.long)
train_size = int(.9*len(data))
train_data = data[:train_size]
val_data = data[train_size:]

def get_batch(split:str)->torch.Tensor:
    dataset = train_data if split == 'train' else val_data
    idx = torch.randint(len(dataset)-tr.CONTEXT_LENGTH, (tr.BATCH_SIZE, )) 
    x = torch.stack([dataset[i:i+tr.CONTEXT_LENGTH] for i in idx])
    y = torch.stack([dataset[i+1:i+tr.CONTEXT_LENGTH+1] for i in idx])
    return x, y

model = tr.Transformer(vocab_size)

@torch.no_grad()
def estimate_loss():
    losses = {} 
    model.eval()
    for split in ['train', 'val']:
        ls = torch.zeros(tr.EVAL_ITR).to(tr.DEVICE)
        for k in range(tr.EVAL_ITR):
            x, y = get_batch(split)
            x, y = x.to(tr.DEVICE), y.to(tr.DEVICE)
            _, loss = model(x, y)
            ls[k] = loss.item()
        losses[split] = ls.mean().item()
    model.train()
    return losses

#training loop
model = model.to(tr.DEVICE).train()
optimizer = tr.optim(model.parameters(), lr= tr.LR)
for i in range(tr.EPOCHS):
    if i % tr.EVAL_INTERVAL == 0 :
        losses = estimate_loss()
        print(f'step: {i}{" "*(3-len(str(i)))} | training loss:{losses["train"]:.4f} | eval loss:{losses["val"]:.4f}')
    
    x, y = get_batch('train')
    x, y = x.to(tr.DEVICE), y.to(tr.DEVICE)
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
losses = estimate_loss()
print(f'step: {i}{" "*(4-len(str(i)))} | training loss:{losses["train"]:.4f} | eval loss:{losses["val"]:.4f}')

ip = torch.tensor([encode('hello cesar ')], device=tr.DEVICE)
print(decode(model.generate(ip, 500)[0].tolist()))
