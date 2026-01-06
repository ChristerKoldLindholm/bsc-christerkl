import numpy as np
import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        # Encoder. 
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Decoder.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x) # x = (t * M), z = (t, latent_dim)
        # Reconstruction.
        x_hat = self.decoder(z) # (t, M)
        return x_hat, z 
    
class LogMelDownsampledDataset(torch.utils.data.Dataset):
    def __init__(self, paths, key="feature_ds"):
        self.paths = list(paths)
        self.key = key

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        arr = np.load(path)[self.key] # x = (t, M).
        x = torch.from_numpy(arr).float()
        x = x.flatten() # (t * M)
        return x, str(path)