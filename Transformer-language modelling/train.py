import torch
import pkg.transformer as tr

# path to training data
def path()->str:
    """
    Returns the path of the TinyShakespeare.txt file.

    :return: A string representing the path of the TinyShakespeare.txt file.
    """
    return "TinyShakesphere.txt"

# read training data
with open(path(), 'r', encoding='utf-8') as f:
    text = f.read()

# build vocabulary
vocab = list(set(text))
vocab_size = len(vocab)

# encode input string to index
def encode(text):
    """
    Encodes the given text using a character-to-index mapping.

    Args:
        text (str): The text to be encoded.

    Returns:
        list: A list of integers representing the encoded text.
    """
    chartoi = {char:i for i, char in enumerate(vocab)}
    return [chartoi[char] for char in text]

# decode index to string
def decode(idx_lst):
    """
    Decode a list of indices into a string using the given vocabulary.

    Args:
    - idx_lst (list): A list of integer indices representing characters.

    Returns:
    - str: The decoded string.
    """
    itochar = {i:char for i, char in enumerate(vocab)}
    return ''.join([itochar[i] for i in idx_lst])

# split data and build training and validation dataset
data = torch.tensor(encode(text), dtype=torch.long)
train_size = int(.9*len(data))
train_data = data[:train_size]
val_data = data[train_size:]

# get random batch
def get_batch(split:str)->torch.Tensor:
    """
    Generate a batch of data for training or validation.

    Args:
        split (str): The split of the data to generate the batch for. It can be either 'train' or 'val'.

    Returns:
        torch.Tensor: The input data batch.
        torch.Tensor: The target data batch.
    """
    dataset = train_data if split == 'train' else val_data
    idx = torch.randint(len(dataset)-tr.CONTEXT_LENGTH, (tr.BATCH_SIZE, )) 
    x = torch.stack([dataset[i:i+tr.CONTEXT_LENGTH] for i in idx])
    y = torch.stack([dataset[i+1:i+tr.CONTEXT_LENGTH+1] for i in idx])
    return x, y

# define model
model = tr.Transformer(vocab_size)

# unbiased estimate of loss
@torch.no_grad()
def estimate_loss():
    """
    Estimates the loss of the model on the training and validation sets.

    Returns:
        losses (dict): A dictionary containing the average loss for each split.
            The keys are 'train' and 'val', and the values are floats.
    """

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

# training loop
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

# generate text
ip = torch.tensor([encode('hello cesar ')], device=tr.DEVICE)
print(decode(model.generate(ip, 500)[0].tolist()))
