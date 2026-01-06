# (1) Loads the full feature set from a Slurm folder.
# (2) Irrelevant log-mel bins are removed.
# (3) Downsamples the spectrogram feature vectors to uniform length.
# (4) Flattens the spectrogram feature vectors.
# (5) Performs PCA, applies elbow method, k-means clustering on the full feature set.
# (6) Save the results.

import argparse
from datetime import datetime
import multiprocessing
import numpy as np 
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import sys
from time import perf_counter

# Custom libraries. 
utils_path = Path.cwd().parents[1]
sys.path.insert(0, str(utils_path))
import feature_utils as futils

class Checkpoints:
    def __init__(self): self.t = perf_counter()
    def tick(self, label): 
        now = perf_counter()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {label}: {now - self.t:.2f}s", flush=True)
        self.t = now

def slurm_cpus():
    try:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", "").strip() or 0) or multiprocessing.cpu_count()
    except Exception:
        return multiprocessing.cpu_count()

def kmeans_score_curves(X, k_range=range(2, 16), batch_size=8192, sample_size=10_000, random_state=104):
    n = len(X)
    rng = np.random.RandomState(random_state)
    
    try:
        cpus = slurm_cpus()
    except NameError:
        cpus = 1
    b = min(batch_size, 256 * cpus)  # Scale with allocated CPUs if under Slurm.

    if sample_size and n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X_sample = X[idx]
    else: 
        idx = None
        X_sample = X

    inertias = []
    silhouette_scores = []
    b = min(batch_size, 256 * cpus)  # Scale with allocated CPUs if under Slurm.

    for k in k_range:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            init="auto",
            n_init=10,
            batch_size=b,
            max_iter=100,
            random_state=random_state,
            verbose=0,
        )
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

        labels_full = kmeans.labels_
        labels_s = labels_full[idx] if idx is not None else labels_full

        # Silhouette scores.
        if len(np.unique(labels_s)) > 1:
            try:
                s = silhouette_score(X_sample, labels_s, metric="euclidean")
            except Exception:
                s = np.nan
        else:
            s = np.nan        
        silhouette_scores.append(s)

    return np.asarray(inertias, dtype=np.float32), np.asarray(silhouette_scores, dtype=np.float32)

def kmeans_silhouette_scores(X, k_range=range(2, 16), sample_size=10000, random_state=0):
    scores = []
    for k in k_range:
        km = MiniBatchKMeans(k, n_init='auto', random_state=random_state).fit(X)
        labels = km.labels_
        # Subsample if 
        if sample_size and len(X) > sample_size:
            idx = np.random.RandomState(random_state).choice(len(X), sample_size, replace=False)
            s = silhouette_score(X[idx], labels[idx])
        else:
            s = silhouette_score(X, labels)
        scores.append(s)
    return np.array(scores)
    
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
            s = futils.downsample_time_avgpool_from_db(s, T_target=8000, ref=1.0) # Returns tensor.
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

    ck = Checkpoints()

    data_path = Path(args.input_root)
    output_path = Path(args.output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    ck.tick("Downsampling")
    # (1) Loads the full feature set from a Slurm folder.
    # (2) Irrelevant log-mel bins are removed.
    # (3) Downsamples the spectrogram feature vectors to uniform length.
    # (4) Flattens the spectrogram feature vectors.
    X, used_paths = files_to_vectors(data_path, n_mels=args.n_mels, key="feature")
    print("X shape:", X.shape)  # (n_files, T_target * (n_mels - 40))

    ck.tick("PCA")
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

    ck.tick("KMeans elbow scores")
    # KMeans
    seed = 104
    inertias, silhouettes = kmeans_score_curves(Z_95, k_range=range(2, 16), batch_size=8192
        , random_state=seed)

    ck.tick("Saving results")
    # (6) Save the PCA and clustering results.
    np.savez_compressed(
        output_path / "kmeans_curve_scores.npz",
        used_paths=np.array([str(p) for p in used_paths]),
        inertias=np.array(inertias),
        silhouettes=np.array(silhouettes),
    )

if __name__ == "__main__":
    main()