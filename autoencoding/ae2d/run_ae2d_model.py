import argparse
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler, random_split

import_path = Path.cwd().parents[1]
os.sys.path.insert(0, str(import_path))
import feature_utils as futils
import utils
import autoencoder_functions as aefuncs

# Standardize batches during model training.
def standardize_batch(x, scaler):
    # Input x = (B, 1, M, t).
    x_np = x.squeeze(1).cpu().numpy() # x_np = (B, M, t).
    B, M, T = x_np.shape
    X = np.transpose(x_np, (0, 2, 1)).reshape(-1, M) # (B * T, M).
    X_std = scaler.transform(X) # (B * T, M).
    x_s = X_std.reshape(B, T, M).transpose(0, 2, 1) # (B, M, t).
    x_s = torch.from_numpy(x_s).unsqueeze(1).to(x.device) # (B, 1, M, t).
    return x_s

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
        print(f"Fitted scaler on {b+1} batches")

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

def main():
    print("Running AE2D program...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory with .npz feature files.")
    ap.add_argument("--aug_data_dir", type=str, default=None, help="Directory with secondary .npz feature files.")
    ap.add_argument("--sep_data_dir", type=str, default=None, help="Directory with secondary .npz feature files.")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    ap.add_argument("--scaler_dir", type=str, default=None, help="Directory to load scaler from.")
    ap.add_argument("--n_epochs", type=int, default=20, help="Number of training epochs.")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    ap.add_argument("--segment_sec", type=float, default=3.0, help="Segment length in seconds.")
    ap.add_argument("--T_target", type=int, default=200, help="Target number of time frames after segmentation.")
    args = ap.parse_args()

    # Paths.
    data_dir = Path(args.data_dir)
    aug_data_dir = Path(args.aug_data_dir) if args.aug_data_dir is not None else None
    sep_data_dir = Path(args.sep_data_dir) if args.sep_data_dir is not None else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configurations.
    device = "cpu"
    key = "feature"

    mels_start, mels_end, n_mels = 9, 61, 128
    mels_used = mels_end - mels_start
    sr = 64000
    hop_length = 512
    fps = sr / hop_length
    t_start, t_end = 25.0, 30.0
    i_start = int(t_start * fps)
    i_end = int(t_end * fps)
    segment_sec, clip_sec = args.segment_sec, 265
    T_target = args.T_target

    # Prepare dataset.
    batch_size = args.batch_size
    rng = np.random.default_rng(seed=104)

    print("Preparing dataset...")
    tracy_paths = sorted(data_dir.glob("*.npz")) 
    aug_paths = sorted(aug_data_dir.glob("*.npz")) if aug_data_dir is not None else []
    sep_paths = sorted(sep_data_dir.glob("*.npz")) if sep_data_dir is not None else []

    n_tracy_files = 4096
    tracy_files = list(rng.choice(tracy_paths, size=min(n_tracy_files, len(tracy_paths)), replace=False))
    n_aug_files = 1024
    aug_files = list(rng.choice(aug_paths, size=min(n_aug_files, len(aug_paths)), replace=False))
    n_sep_files = 1024
    sep_files = list(rng.choice(sep_paths, size=min(n_sep_files, len(sep_paths)), replace=False))

    # tracy_paths.extend([*aug_files, *sep_files])
    # all_files = list(tracy_paths)
    all_files = tracy_files + aug_files + sep_files
    n_files = len(all_files)

    with np.load(all_files[0], mmap_mode="r") as z:
        arr = np.squeeze(z[key])
        if arr.ndim == 3:
            arr = arr[0]
        t_total = arr.shape[-1]

    frames_per_sec = t_total / clip_sec
    segment_frames = int(round(frames_per_sec * segment_sec)) # = (t / 265) * segment_sec.

    n_test_files = min(256, n_files)
    n_train_files = int(n_files * 0.8)
    n_train_files = min(n_train_files, n_files - n_test_files)
    n_val_files = n_files - n_train_files - n_test_files
    n_val_files = max(0, n_val_files)

    train_files = list(rng.choice(all_files, size=n_train_files, replace=False))
    # Include known calls in training set only.
    narwhal_files = [data_dir / "Tracy_6230" / "6230.220917120000.npz"
                     , data_dir / "Tracy_6230" / "6230.221003000000.npz"]
    for p in narwhal_files:
        if p.exists() and p not in train_files:
            train_files.append(p)
    remaining_files = list(set(all_files) - set(train_files))
    val_files = list(rng.choice(remaining_files, size=n_val_files, replace=False))
    remaining_files = list(set(remaining_files) - set(val_files))
    test_files = list(rng.choice(remaining_files, size=n_test_files, replace=False))

    print(f"Total files = {n_files}, train = {len(train_files)}, val = {len(val_files)}, test = {len(test_files)}")

    train_dataset = aefuncs.SegmentLogMel2DDataset(train_files, key=key, mels_start=mels_start, mels_end=mels_end
                                                   , segment_frames=segment_frames, T_target=T_target, add_channel_dim=True)
    val_dataset = aefuncs.SegmentLogMel2DDataset(val_files, key=key, mels_start=mels_start, mels_end=mels_end
                                                 , segment_frames=segment_frames, T_target=T_target, add_channel_dim=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Scaler.
    # scaler = StandardScaler(with_mean=True, with_std=True)
    # for x, _meta in train_loader: # x = (B,1,M,T)
    #     x = x.squeeze(1).cpu().numpy() # (B,M,T)
    #     B, M, T = x.shape
    #     X = np.transpose(x, (0, 2, 1)).reshape(-1, M) # (B*T, M)
    #     scaler.partial_fit(X)
    #     print(f"Scaler fitted on {scaler.n_samples_seen_} samples.")

    # scaler = fit_scaler_streaming(train_loader, device=device, max_batches=100)
    # np.savez_compressed(output_dir / "scaler.npz"
    #         , scaler_mean=scaler.mean_
    #         , scaler_var=scaler.var_
    #         , scaler_scale=scaler.scale_
    #         , n_samples_seen=scaler.n_samples_seen_
    #         )

    print("Loading scaler from", args.scaler_dir)
    scaler = StandardScaler(with_mean=True, with_std=True)
    with np.load(Path(args.scaler_dir) / "scaler.npz") as z:
        scaler.mean_ = z["scaler_mean"]
        scaler.var_ = z["scaler_var"]
        scaler.scale_ = z["scaler_scale"]
        scaler.n_samples_seen_ = z["n_samples_seen"]

    print("Scaler shapes =", scaler.mean_.shape, scaler.scale_.shape)

    data_point, meta = next(iter(train_loader))
    print("data_point.shape =", data_point.shape) # (B, C, M = n_mels, t = segment_frames).
    print("meta: file =", meta[0][0], ", start =", meta[1][0], ", end =", meta[2][0])
    print("Data sample:", data_point[0, :, 0:5, 0:5])

    input_channels = data_point.shape[1]
    lat_dim = 2
    hid_dim = 16
    lr = 1e-3
    # 2D autoencoder: CNN.
    model = aefuncs.AE2D(input_channels=input_channels, latent_dim=lat_dim, hidden_dim=hid_dim).to(device)
    # U-Net autoencoder: input channels equals output channels when reconstructing with MSE loss.
    # model = aefuncs.AEUnet2D(in_channels=input_channels, out_channels=input_channels, base=32, bilinear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    train_losses, val_losses = [], []
    n_epochs = args.n_epochs
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 2
    min_delta = 0.05

    print("Initializing Torch standardizer...")
    standardizer = TorchStandardizer(scaler, device)

    # Save stats.
    np.savez_compressed(output_dir / "model_results.npz"
                , n_files=n_files
                , train_files=[str(p) for p in train_files]
                , n_train_files=len(train_files)
                , val_files=[str(p) for p in val_files]
                , n_val_files=len(val_files)
                , test_files=[str(p) for p in test_files]
                , n_test_files=n_test_files
                , segment_frames=segment_frames
                , T_target=T_target
                , mels_start=mels_start
                , mels_end=mels_end
                , lat_dim=lat_dim
                , hid_dim=hid_dim
                , lr=lr
                , batch_size=batch_size
                , patience=patience
                , scaler_scale=scaler.scale_
                , scaler_mean=scaler.mean_
                , scaler_var=scaler.var_
                )

    for epoch in range(n_epochs):
        print("Starting epoch", epoch + 1)
        # Training.
        model.train()
        running_train_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device) # (B, C, M, T).
            data = standardizer.transform(data)
            optimizer.zero_grad()
            outputs, _ = model(data) 
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * data.size(0)
        train_loss  = running_train_loss / len(train_loader.dataset)

        # Validation.
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x, meta in val_loader:
                x = x.to(device)
                x = standardizer.transform(x)
                x_hat, z = model(x)
                loss = criterion(x_hat, x)

                running_val_loss += loss.item() * x.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        np.savez_compressed(output_dir / "train_stats.npz"
                    , n_epochs=epoch + 1
                    , train_losses=np.array(train_losses)
                    , val_losses=np.array(val_losses)
                    )

        torch.save(model.state_dict(), output_dir / "latest_ae2d_model.pth")
        torch.save(optimizer.state_dict(), output_dir / "latest_ae2d_optimizer.pth")
        print("Saved latest model.")

        print(f"Epoch [{epoch+1}/{n_epochs}], train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
        print("Train losses =", [f"{loss:.4f}" for loss in train_losses])
        print("Val losses =", [f"{loss:.4f}" for loss in val_losses])

        # Early stopping.
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            # Save best model.
            torch.save(model.state_dict(), output_dir / "best_ae2d_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break 
    
if __name__ == "__main__":
    main()