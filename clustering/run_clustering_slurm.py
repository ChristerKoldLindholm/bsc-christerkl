import argparse
import multiprocessing
import numpy as np 
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

def slurm_cpus():
    try:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", "").strip() or 0) or multiprocessing.cpu_count()
    except Exception:
        return multiprocessing.cpu_count()
    
def files_to_vectors(folder: Path, n_mels: int = 128, key: str = "feature", return_paths: bool = True) -> np.ndarray:
    """
    For each .npz, we load the feature array, and build [mu, std] matrix with shape (n_files, 2*n_mels).
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

            F = np.squeeze(z[key])
            if F.ndim == 1:
                raise ValueError(f"{fp} has 1D feature {F.shape}")

            # Ensure shape is (T, n_mels)
            if F.shape[0] == n_mels and F.shape[1] != n_mels:
                F = F.T
            elif F.shape[1] == n_mels:
                pass
            else:
                raise ValueError(f"{fp}: unexpected shape {F.shape}; neither axis equals n_mels={n_mels}")

            # Handle potential NaNs/Infs defensively (optional)
            if not np.isfinite(F).all():
                F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

            mu = F.mean(axis=0)              # (n_mels,)
            sd = F.std(axis=0)               # (n_mels,)
            vec = np.concatenate([mu, sd]).astype(np.float32, copy=False)

            rows.append(vec[None, :])
            used_paths.append(fp)

    if not rows:
        raise ValueError(f"No arrays with key '{key}' found under {folder}")

    X = np.vstack(rows)  # (n_files, 2*n_mels)
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

    # Load per-file [mu|sd] blocks.
    X, used_paths = files_to_vectors(data_path, n_mels=args.n_mels, key="feature")
    print("X shape:", X.shape)  # (n_files, 2*n_mels)

    # Split into mean and std blocks.
    n = args.n_mels
    M = X[:, :n]
    S = X[:, n:2*n]

    # Standardize each block separately
    M_zs = StandardScaler(with_mean=True, with_std=True).fit_transform(M)
    S_zs = StandardScaler(with_mean=True, with_std=True).fit_transform(S)
    X_zs = np.concatenate((M_zs, S_zs), axis=1)

    # PCA: let sklearn choose the solver automatically
    pca = PCA(n_components=None, svd_solver="auto", random_state=104)
    Z = pca.fit_transform(X_zs)

    # Choose dimensionality for desired cumulative variance
    var_lim = float(args.var_thresh)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    idx = int(1 + np.argmax(cumsum >= var_lim)) if np.any(cumsum >= var_lim) else Z.shape[1]
    print(f"Number of components for {var_lim*100:.1f}% variance: {idx}")
    Z_95 = Z[:, :idx]

    # KMeans
    seed = 104
    k = int(args.clusters)
    b = 256 * slurm_cpus()  # Scale with allocated CPUs if under Slurm.
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
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
        sil = silhouette_score(Z_95, labels) if len(np.unique(labels)) > 1 and Z_95.shape[0] > k else np.nan
    except Exception:
        sil = np.nan

    np.savez_compressed(
        output_path / "feats_pca_kmeans.npz",
        X=X,
        used_paths=np.array([str(p) for p in used_paths]),
        Z=Z.astype(np.float32, copy=False),
        Z_95=Z_95.astype(np.float32, copy=False),
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