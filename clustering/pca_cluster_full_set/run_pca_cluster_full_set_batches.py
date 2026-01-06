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
from sklearn.decomposition import PCA, IncrementalPCA
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
    
def files_to_vectors_batch(folder: Path, n_mels: int=128, batch_files: int=64, drop_mels: int=40, key: str="feature", return_paths: bool=True):
    """
    For each .npz, load the feature array s_i, flatten s_i, then build full matrix S for PCA and k-means.
    """
    files = sorted(folder.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")

    batch, paths = [], []
    for fp in files:
        with np.load(fp, mmap_mode="r") as z:
            if key not in z.files:
                continue
            s = np.squeeze(z[key])
            if s.ndim == 1:
                continue 
            # Ensure shape is (T, n_mels)
            if s.shape[0] == n_mels and s.shape[1] != n_mels:
                s = s.T
            elif s.shape[1] == n_mels:
                pass
            else:
                continue 
            if not np.isfinite(s).all():
                s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

            # Remove irrelevant low-frequency bins (first 40 mel bins).
            s = s[:, drop_mels:] # (t, M=120-40). 
            s = futils.downsample_time_avgpool_from_db(s, T_target=8000, ref=1.0) # Returns tensor.
            v = s.detach().cpu().numpy().astype(np.float32, copy=False).ravel()  # Flattened feature vector: (8000*88, )

            batch.append(v[None, :])
            paths.append(fp)

        if len(batch) == batch_files:
            Xb = np.vstack(batch).astype(np.float32, copy=False)
            yield Xb, paths 
            batch, paths = [], []

    if batch:
        Xb = np.vstack(batch).astype(np.float32, copy=False)
        yield Xb, paths

def fit_streaming_pca_kmeans(data_path: Path,
                             n_components: int = 256,
                             k: int = 8,
                             batch_files: int = 512,
                             sample_size: int = 8000,
                             random_state: int = 104):

    # — pass 1: fit scaler on streaming batches —
    scaler = StandardScaler(with_mean=True, with_std=True)
    used_paths = []
    for Xb, paths in files_to_vectors_batch(data_path, batch_files=batch_files):
        scaler.partial_fit(Xb)
        used_paths.extend(paths)

    # — pass 2: fit IncrementalPCA on standardized batches —
    ipca = IncrementalPCA(n_components=n_components, batch_size=None)
    for Xb, _ in files_to_vectors_batch(data_path, batch_files=batch_files):
        Xb_std = scaler.transform(Xb)
        ipca.partial_fit(Xb_std)

    # — pass 3: train MiniBatchKMeans while transforming —
    # also keep a small sample of Z for silhouette
    rng = np.random.default_rng(random_state)
    keep = []  # small reservoir of Z
    kmeans = MiniBatchKMeans(
        n_clusters=k, init="k-means++", n_init="auto",
        batch_size=min(8192, 256 * slurm_cpus()),
        max_iter=100, random_state=random_state
    )

    for Xb, _ in files_to_vectors_batch(data_path, batch_files=batch_files):
        Zb = ipca.transform(scaler.transform(Xb)).astype(np.float32, copy=False)
        kmeans.partial_fit(Zb)

        # Reserve sample for silhouette scores.
        if sample_size > 0:
            # add all then downsample (simple & fine for moderate sample_size)
            keep.append(Zb)
            total_kept = sum(len(a) for a in keep)
            if total_kept > 2 * sample_size:
                Ztmp = np.vstack(keep)
                idx = rng.choice(len(Ztmp), sample_size, replace=False)
                keep = [Ztmp[idx]]

    sil = np.nan
    if keep:
        Zs = np.vstack(keep)
        if len(Zs) > k:
            labels_s = kmeans.predict(Zs)
            if len(np.unique(labels_s)) > 1:
                sil = float(silhouette_score(Zs, labels_s))

    return scaler, ipca, kmeans, sil, used_paths

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

    # (5) Performs PCA and k-means clustering on the full feature set.
    N_COMPONENTS_CAP = 256
    N_BATCH_FILES = 512
    K = int(args.clusters)

    scaler, ipca, kmeans, sil, used_paths = fit_streaming_pca_kmeans(data_path=data_path, n_components=N_COMPONENTS_CAP
        , k=K, batch_files=N_BATCH_FILES, sample_size=8_000, random_state=104)

    # (6) Save the PCA and clustering results.
    np.savez_compressed(
        output_path / "pca_kmeans_batches.npz",
        used_paths=np.array([str(p) for p in used_paths]),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        ipca_components=ipca.components_.astype(np.float32),
        ipca_mean=ipca.mean_.astype(np.float32),
        ipca_explained_variance=ipca.explained_variance_.astype(np.float32),
        ipca_explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32),
        kmeans_centers=kmeans.cluster_centers_.astype(np.float32),
        kmeans_inertia=float(kmeans.inertia_),
        silhouette=float(sil),
    )

if __name__ == "__main__":
    main()