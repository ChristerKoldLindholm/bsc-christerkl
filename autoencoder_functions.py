import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

import feature_utils as futils
import utils
from configs import get_specgram_config

# ==================================================
# Autoencoder model: flattened log-mel spectrogram segments.
class AE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        # Encoder: R in (M, t_seg) -> R in (latent_dim). 
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Decoder. R in (latent_dim) -> R in (M, T).
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, x):
        # x = (B, D) = (M, t_seg).
        z = self.encoder(x) # -> (B, latent_dim) # x = (t * M), z = (t, latent_dim)
        # Reconstruction.
        x_hat = self.decoder(z) # -> (B, D) # (t, M)
        return x_hat, z 

class SegmentLogMelDataset(torch.utils.data.Dataset):
    def __init__(self, paths, key="feature", mels_start=9, mels_end=60
                 , segment_frames: int = 600, max_frames: int | None = None, T_target: int = 200,):
        self.paths = list(paths)
        self.key = key
        self.mels_start = mels_start
        self.mels_end = mels_end
        self.segment_frames = segment_frames
        self.max_frames = max_frames
        self.T_target = T_target

        self.index = []
        self._build_index()

        self.M_used = self.mels_end - self.mels_start
        self.input_dim = self.M_used * T_target

    def _build_index(self):
        for path in self.paths:
            with np.load(path, mmap_mode="r") as z:
                if self.key not in z.files:
                    continue
                arr = z[self.key]  # (M, t) or (1, M, t) -> (M, t).
                if arr.ndim == 3:
                    arr = arr[0]
                arr = arr[self.mels_start:self.mels_end, :]  # (M_used, t).
                t = arr.shape[1]
                if self.max_frames is not None:
                    t = min(t, self.max_frames)

                for start in range(0, t - self.segment_frames +1, self.segment_frames):
                    end = start + self.segment_frames
                    self.index.append((path, start, end))

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        path, start, end = self.index[idx]
        with np.load(path) as z:
            arr = np.squeeze(z[self.key])
            if arr.ndim == 3:
                arr = arr[0]
            seg = arr[self.mels_start:self.mels_end, start:end]  # (M_used, segment_frames).
        
        seg_ds = futils.downsample_time_avgpool_from_db(seg, T_target=self.T_target, ref=1.0).numpy()  # (M_used, T_target).

        x = torch.from_numpy(seg_ds).flatten()  # (M_used * T_target).
        meta = (str(path), start, end)

        return x, meta

@torch.no_grad()
def fit_scaler_streaming(train_loader, device="cpu", max_batches=None):
    # Running per-feature mean/variance for flattened (B*T) samples.
    n = 0
    mean = None
    M2 = None

    for b, (x, _meta) in enumerate(train_loader): 
        if max_batches is not None and b >= max_batches:
            break
        # x: (B,1,M,T) or (B,C,M,T).
        x = x.to(device, non_blocking=True)
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0] # (B,M,T).
        else:
            # if C>1.
            x = x[:, 0]

        # Compute per M feature mean/var for batch. Flatten time to stack the M dimensions.
        X = x.permute(0, 2, 1).reshape(-1, x.shape[1]) # (B,M,T) -> (B*T, M).

        batch_n = X.shape[0]
        batch_mean = X.mean(dim=0)
        batch_var = X.var(dim=0, unbiased=False)

        if mean is None:
            mean = batch_mean
            M2 = batch_var * batch_n
            n = batch_n
        else:
            delta = batch_mean - mean
            tot = n + batch_n
            mean = mean + delta * (batch_n / tot)
            M2 = M2 + batch_var * batch_n + delta.pow(2) * (n * batch_n / tot)
            n = tot

    var = M2 / n
    scale = torch.sqrt(var + 0.0)

    # Populate a StandardScaler-like object.
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.mean_ = mean.cpu().numpy()
    scaler.var_ = var.cpu().numpy()
    scaler.scale_ = scale.cpu().numpy()
    scaler.n_samples_seen_ = np.array([n], dtype=np.int64)  # sklearn uses array-like sometimes
    return scaler

class TorchStandardizer:
    def __init__(self, scaler, device, eps: float = 1e-8, dtype=torch.float32):
        # Both shapes = (M,).
        mean = torch.as_tensor(scaler.mean_, dtype=dtype, device=device)
        std  = torch.as_tensor(scaler.scale_, dtype=dtype, device=device)
        # Reshape for broadcasting over (B, C, M, T). -> (1, 1, M, 1)
        self.mean = mean.view(1, 1, -1, 1)
        self.std  = std.view(1, 1, -1, 1)
        self.eps = eps

    @torch.no_grad()
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, M, T).
        return (x - self.mean) / (self.std + self.eps)

    @torch.no_grad()
    def inverse_transform(self, xz: torch.Tensor) -> torch.Tensor:
        return xz * (self.std + self.eps) + self.mean
    
def fit_running_standardizer(Z, n, mean, M2):
    if Z.ndim != 2:
        raise ValueError("Z shape is not (B, D).")

    B = Z.shape[0]
    batch_mean = Z.mean(dim=0)
    batch_var = Z.var(dim=0, unbiased=False)
    batch_M2 = batch_var * B # M2 is the running variance sum.
    
    if mean is None:
        return B, batch_mean, batch_M2
    delta = batch_mean - mean
    tot = n + B
    
    mean_new = mean + delta * (B / tot)
    M2_new = M2 + batch_M2 + delta.pow(2) * (n * B / tot)

    return tot, mean_new, M2_new

def finalize_running_standardizer(n, mean, M2, eps=1e-8):
    var = M2 / n
    scale = torch.sqrt(var + eps)
    return mean, scale

    
# ==================================================
# Autoencoder with dimensions.
class AE2D(nn.Module):
    def __init__(self, input_channels:int, latent_dim:int=8, hidden_dim:int=64):
        super().__init__()

        # Encoder: Conv2D layers (M, T) -> (B, latent_dim, M//8, T//8).
        self.encoder = nn.Sequential(
            # (B, M, T) -> (B, hidden_dim, M//2, T//2).
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (B, hidden_dim, M//2, T//2) -> (B, latent_dim, M//4, T//4).
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (B, latent_dim, M//4, T//4) -> (B, latent_dim, M//8, T//8).
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Decoder: ConvTranspose2D to reconstruct (M, T).
        # (B, latent_dim, M//8, T//8) -> (B, 1, M, T).
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1, output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_channels, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
        )

        # Forward.
    def forward(self, x):
        # x = (B, C, M, T).
        z = self.encoder(x)  # -> (B, latent_dim, M//8, T//8).
        x_hat = self.decoder(z)  # -> (B, C, M, T).
        return x_hat, z
    
class SegmentLogMel2DDataset(torch.utils.data.Dataset):
    """
    Returns:
    x: (C=1, M_used, T_target) tensor.
    meta: (path, start, end).
    """

    def __init__(self, paths, key="feature", mels_start=9, mels_end=60
                 , segment_frames: int=600, max_frames: int | None=None, T_target: int=200
                 , add_channel_dim: bool=True):
        
        self.paths = list(paths)
        self.key = key
        self.mels_start = mels_start
        self.mels_end = mels_end
        self.segment_frames = segment_frames
        self.max_frames = max_frames
        self.T_target = T_target
        self.add_channel_dim = add_channel_dim

        self.index = []
        self._build_index()

        self.M_used = self.mels_end - self.mels_start
        self.input_shape = (1 if add_channel_dim else self.M_used, self.M_used, T_target) \
            if add_channel_dim else (self.M_used, T_target)
        
    def _build_index(self):
        for path in self.paths:
            with np.load(path, mmap_mode="r") as z:
                if self.key not in z.files:
                    continue 
                
                arr = np.squeeze(z[self.key]) # (M, t) or (1, M, t) -> (M, t).
                if arr.ndim == 3:
                    arr = arr[0]
                if arr.ndim != 2:
                    continue

                arr = arr[self.mels_start:self.mels_end, :]  # (M_used, t).
                t = arr.shape[1]
                if self.max_frames is not None:
                    t = min(t, self.max_frames)

                step = self.segment_frames
                for start in range(0, t - self.segment_frames + 1, step):
                    end = start + self.segment_frames
                    self.index.append((path, start, end))

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        path, start, end = self.index[idx]
        with np.load(path, mmap_mode="r") as z:
            arr = np.squeeze(z[self.key])
            if arr.ndim == 3:
                arr = arr[0]
            seg = arr[self.mels_start:self.mels_end, start:end]  # (M_used, segment_frames).

        seg_ds = futils.downsample_time_avgpool_from_db(seg, T_target=self.T_target, ref=1.0) # (M_used, T_target).
        # x = torch.from_numpy(seg_ds) # (M_used, T_target).
        if self.add_channel_dim:
            x = seg_ds.unsqueeze(0)  # (1, M_used, T_target).

        meta = (str(path), start, end)

        return x, meta
    
# ==================================================
# U-Net autoencoder with 2D convs.
# See reference implementation: 
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) x 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Downscale(nn.Module):
    """Downscaling with maxpool then double conv."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )
    
    def forward(self, x):
        return self.net(x)
    
class Upscale(nn.Module):
    """Upscale then double conv."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, bilinear: bool=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad x to match spatial dimensions with odd sizes.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)
    
class UNet2D(nn.Module):
    """U-Net for (B, C, M, T) inputs."""
    def __init__(self, in_channels: int, out_channels: int, base: int=32, bilinear: bool=True):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Downscale(base, base*2)
        self.down2 = Downscale(base*2, base*4)
        self.down3 = Downscale(base*4, base*8)

        self.bot = DoubleConv(base*8, base*16) # Bottleneck: most spatially compressed
        # and minimum resolution. The latent space or feature map.

        self.up1 = Upscale(base*16, base*8, base*8, bilinear=bilinear)
        self.up2 = Upscale(base*8, base*4, base*4, bilinear=bilinear)
        self.up3 = Upscale(base*4, base*2, base*2, bilinear=bilinear)
        self.up4 = Upscale(base*2, base, base, bilinear=bilinear)

        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        # x = (B, C, M, T).
        x1 = self.inc(x) # -> (B, base, M, T).
        x2 = self.down1(x1) # -> (B, base*2, M/2, T/2).
        x3 = self.down2(x2) # -> (B, base*4, M/4, T/4).
        x4 = self.down3(x3) # -> (B, base*8, M/8, T/8).

        x_bot = self.bot(x4) # -> (B, base*16, M/8, T/8).

        x = self.up1(x_bot, x4) # -> (B, base*8, M/4, T/4).
        x = self.up2(x, x3) # -> 4, 2.
        x = self.up3(x, x2) # -> 2, 1.
        x = self.up4(x, x1) # -> Base.

        x_out = self.outc(x) # -> (B, out_channels, M, T).

        return x_out, x_bot