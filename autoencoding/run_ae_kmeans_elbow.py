#!/usr/bin/env python3
"""
Run k-means with elbow method on autoencoder latent features for multiple latent dims.
"""
import argparse
import numpy as np
import multiprocessing
import os
from pathlib import Path

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import ae_functions as aefuncs

def slurm_cpus():
    try:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", "").strip() or 0) or multiprocessing.cpu_count()
    except Exception:
        return multiprocessing.cpu_count()
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=str, required=True, help="Folder containing .npz feature files (with feature_ds).")
    ap.add_argument("--saved-pca", type=str, required=True, help="Path to prior pca_kmeans_batches.npz with scaler_mean and scaler_scale.")
    ap.add_argument("--output-root", type=str, required=True, help="Folder to write outputs.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--sample-files", type=int, default=0, help="If >0, subsample this many files for elbow & PCA plots.")
    ap.add_argument("--sample-seed", type=int, default=104)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_root = Path(args.input_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Prepare dataset. 
    dataset = aefuncs.LogMelDownsampledDataset(input_root.rglob("*.npz"), key="feature_ds")
    n = len(dataset)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    g = torch.Generator().manual_seed(args.sample_seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=g)

    num_workers = min(8, slurm_cpus())
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # Dimensions.
    x0, _ = next(iter(train_loader))
    input_dim = x0.shape[1]

    # Load saved scaler stats from PCA npz.
    saved = np.load(args.saved_pca, allow_pickle=False)
    scaler_mean = saved["scaler_mean"].astype(np.float32)
    scaler_scale = saved["scaler_scale"].astype(np.float32)
    # Avoid division by zero.
    scaler_scale[scaler_scale == 0.0] = 1.0

    scaler_mean_t = torch.from_numpy(scaler_mean).to(device)
    scaler_scale_t = torch.from_numpy(scaler_scale).to(device)

    def standardize_batch(x: torch.Tensor) -> torch.Tensor:
        return (x - scaler_mean_t) / scaler_scale_t

    # Model.
    model = aefuncs.AE(input_dim=input_dim, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_losses, val_losses = [], []
    best_val = float("inf")
    epochs_no_improve = 0

    # Training loop with early stopping.
    for epoch in range(1, args.max_epochs + 1):
        # Train
        model.train()
        running_train = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_std = standardize_batch(x)

            optimizer.zero_grad()
            x_hat_std, z = model(x_std)
            loss = F.mse_loss(x_hat_std, x_std)
            loss.backward()
            optimizer.step()

            running_train += loss.item() * x.size(0)

        train_loss = running_train / len(train_loader.dataset)

        # Validation
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_std = standardize_batch(x)
                x_hat_std, z = model(x_std)
                loss = F.mse_loss(x_hat_std, x_std)
                running_val += loss.item() * x.size(0)

        val_loss = running_val / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch}/{args.max_epochs}] "
              f"Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        # Early stopping.
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_root / "ae_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"No improvement for {args.patience} epochs, stopping.")
                break

    # Save training curves and config.
    np.savez_compressed(
        out_root / "ae_metrics.npz",
        train_losses=np.array(train_losses, dtype=np.float32),
        val_losses=np.array(val_losses, dtype=np.float32),
        best_val=float(best_val),
        input_dim=int(input_dim),
        latent_dim=int(args.latent_dim),
        hidden_dim=int(args.hidden_dim),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    # Encode data to latent space Z.
    full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # Load best model.
    model.load_state_dict(torch.load(out_root / "ae_best.pt", map_location=device))
    model.eval()

    Z_list = []
    paths_list = []
    with torch.no_grad():
        for x, paths in full_loader:
            x = x.to(device)
            x_std = standardize_batch(x)
            _, z = model(x_std)      # z = (B, latent_dim)
            Z_list.append(z.cpu().numpy())
            paths_list.extend(list(paths))

    Z = np.vstack(Z_list).astype(np.float32)   # (N, latent_dim)
    paths_arr = np.array(paths_list, dtype=object)

    np.savez_compressed(
        out_root / "ae_latent.npz",
        Z=Z,
        paths=paths_arr,
    )

    # Subsample for elbow visualization.
    rng = np.random.default_rng(args.sample_seed)
    if args.sample_files > 0 and len(Z) > args.sample_files:
        idx = rng.choice(len(Z), size=args.sample_files, replace=False)
        Z_elbow = Z[idx]
    else:
        idx = np.arange(len(Z))
        Z_elbow = Z

    np.save(out_root / "sample_indices.npy", idx.astype(np.int32))

    # k-means elbow on latent space.
    ks = list(range(args.k_min, args.k_max + 1))
    inertias = []
    silhouettes = []

    for k in ks:
        km = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init="auto",
            batch_size=min(1024, args.batch_size * slurm_cpus()),
            max_iter=100,
            random_state=args.sample_seed,
        )
        km.fit(Z_elbow)
        inertias.append(float(km.inertia_))

        labels = km.labels_
        if len(np.unique(labels)) > 1:
            sil = float(silhouette_score(Z_elbow, labels))
        else:
            sil = np.nan
        silhouettes.append(sil)

        # Save centroids in latent space.
        np.save(out_root / f"kmeans_k{k:02d}_centers_latent.npy",
                km.cluster_centers_.astype(np.float32, copy=False))

    np.savez_compressed(
        out_root / "ae_elbow_latent.npz",
        k_values=np.array(ks, dtype=np.int32),
        inertias=np.array(inertias, dtype=np.float64),
        silhouettes=np.array(silhouettes, dtype=np.float64),
    )

if __name__ == "__main__":
    main()