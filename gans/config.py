# import
import torch


path = "path-to-dataset"


# Hyperparameters (need to be tuned)
batch_size = 64
epochs = 1000
latent_dim = 100
lrD = 1e-4
lrG = 1e-4
optimD = torch.optim.AdamW
optimG = torch.optim.AdamW
criterionD = torch.nn.BCELoss()
criterionG = torch.nn.BCELoss()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

