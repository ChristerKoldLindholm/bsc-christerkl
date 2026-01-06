# (1) Loads the full feature set from a Slurm folder.
# (2) Irrelevant log-mel bins are removed.
# (3) Downsamples the spectrogram feature vectors to uniform length.
# (4) Flattens the spectrogram feature vectors.
# (5) Performs PCA and k-means clustering on the full feature set.
# (6) Save the results.

import argparse
import multiprocessing
import numpy as np 
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import sys
# Custom libraries. 
utils_path = Path.cwd().parents[1]
sys.path.insert(0, str(utils_path))
import feature_utils as futils

def slurm_cpus():
    try:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", "").strip() or 0) or multiprocessing.cpu_count()
    except Exception:
        return multiprocessing.cpu_count()
    
def files_to_vectors(folder: Path, n_mels: int = 128, drop_mels: int = 40, key: str = "feature", return_paths: bool = True) -> np.ndarray:
    """
    For each .npz, load the feature array s_i, flatten s_i, then build full matrix S for PCA and k-means.
    """
    files = sorted(folder.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")

    rows = []
    used_paths = []
    for fp in files:
        with np.load(fp, mmap_mode="r") as z:
            if key not in z.files:
                continue

            s = np.squeeze(z[key])
            if s.ndim == 1:
                raise ValueError(f"{fp} has 1D feature {s.shape}")

            # Ensure shape is (T, n_mels)
            if s.shape[0] == n_mels and s.shape[1] != n_mels:
                s = s.T
            elif s.shape[1] == n_mels:
                pass
            else:
                raise ValueError(f"{fp}: unexpected shape {s.shape}; neither axis equals n_mels={n_mels}")

            # Handle potential NaNs/Infs.
            if not np.isfinite(s).all():
                s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

            # Remove irrelevant low-frequency bins (first 40 mel bins).
            s = s[:, drop_mels:] # (t, M=120-40). 
            s = futils.downsample_time_avgpool_from_db(s, T_target=8000, ref=1.0)
            # Move to CPU.
            if hasattr(s, "detach"):
                s = s.detach().cpu()
            s = s.numpy() # (8000, 88)
            s = s.flatten()  # Flattened feature vector: (8000*88, )

            rows.append(s[None, :])
            used_paths.append(fp)

    if not rows:
        raise ValueError(f"No arrays with key '{key}' found under {folder}")

    X = np.vstack(rows).astype(np.float32, copy=False)  # (N files, )
    return (X, used_paths) if return_paths else X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=str, required=True, help="Folder containing .npz feature files.")
    ap.add_argument("--output-root", type=str, required=True, help="Folder to write outputs.")
    ap.add_argument("--n-mels", type=int, default=128, help="Feature dimension expected per frame.")
    ap.add_argument("--var-thresh", type=float, default=0.95, help="Cumulative variance threshold for PCA.")
    ap.add_argument("--clusters", type=int, default=8, help="Number of KMeans clusters.")
    args = ap.parse_args()

    data_path = Path(args.input_root)
    output_path = Path(args.output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    # (1) Loads the full feature set from a Slurm folder.
    # (2) Irrelevant log-mel bins are removed.
    # (3) Downsamples the spectrogram feature vectors to uniform length.
    # (4) Flattens the spectrogram feature vectors.
    X, used_paths = files_to_vectors(data_path, n_mels=args.n_mels, key="feature")
    print("X shape:", X.shape)  # (n_files, T_target * (n_mels - 40))

    # (5) Performs PCA and k-means clustering on the full feature set.
    # Standardize flattened spectrogram vectors.
    scaler = StandardScaler(with_mean=True, with_std=True)
    S_l2 = scaler.fit_transform(X)

    # PCA: let sklearn choose the solver automatically
    pca = PCA(n_components=None, svd_solver="auto", random_state=104)
    Z = pca.fit_transform(S_l2)

    # Choose dimensionality for desired cumulative variance
    var_lim = float(args.var_thresh)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    idx = int(1 + np.argmax(cumsum >= var_lim)) if np.any(cumsum >= var_lim) else Z.shape[1]
    print(f"Number of components for {var_lim*100:.1f}% variance: {idx}")
    Z_95 = Z[:, :idx]

    # KMeans
    seed = 104
    k = int(args.clusters)
    b = min(8192, 256 * slurm_cpus())  # Scale with allocated CPUs if under Slurm.
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        init="auto",
        n_init=10,
        batch_size=b,
        max_iter=100,
        random_state=seed,
        verbose=0,
    )
    kmeans.fit(Z_95)

    labels = kmeans.labels_
    inertia = kmeans.inertia_
    # Silhouette requires at least 2 clusters and >1 sample per cluster.
    try:
        if len(np.unique(labels)) > 1 and Z_95.shape[0] > k:
            m = min (10_000, Z_95.shape[0])
            idx_s = np.random.RandomState(104).choice(Z_95.shape[0], size=m, replace=False)
            sil = silhouette_score(Z_95[idx_s], labels[idx_s])
        else:
            sil = np.nan
    except Exception:
        sil = np.nan

    # (6) Save the PCA and clustering results.
    np.savez_compressed(
        output_path / "feats_pca_kmeans.npz",
        X=X,
        used_paths=np.array([str(p) for p in used_paths]),
        Z=Z.astype(np.float32, copy=False),
        Z_95=Z_95.astype(np.float32, copy=False),
        scaler_mean=scaler.mean_.astype(np.float32, copy=False),
        scaler_scale=scaler.scale_.astype(np.float32, copy=False),
        pca_components=pca.components_.astype(np.float32, copy=False),
        pca_explained_variance=pca.explained_variance_.astype(np.float32, copy=False),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32, copy=False),
        kmeans_labels=labels,
        kmeans_centers=kmeans.cluster_centers_.astype(np.float32, copy=False),
        kmeans_inertia=float(inertia),
        kmeans_silhouette=float(sil),
    )

if __name__ == "__main__":
    main()