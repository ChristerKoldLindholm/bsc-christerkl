# (1) Loads the full feature set from a Slurm folder.
# (2) Irrelevant log-mel bins are removed.
# (3) Downsamples the spectrogram feature vectors to uniform length.
# (4) Flattens the spectrogram feature vectors.
# (5) Performs PCA and k-means clustering on the full feature set.
# (6) Save the results.

import argparse
from collections import defaultdict
import heapq
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
    
def preprocess_feature_array(s, n_mels=128, drop_mels=40):
    """
    s: numpy array from .npz file, shape (t, n_mels).
    Returns:
      v: flattened, downsampled float32 vector (shape D).
      s_ds: downsampled 2D array (T_target, n_mels - drop_mels) as float32.
    """
    if s.ndim == 1:
        return None, None
    # Ensure shape is (T, n_mels)
    if s.shape[0] == n_mels and s.shape[1] != n_mels:
        s = s.T
    elif s.shape[1] == n_mels:
        pass
    else:
        return None, None
    if not np.isfinite(s).all():
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    s = s[:, drop_mels:]
    s_t = futils.downsample_time_avgpool_from_db(s, T_target=8000, ref=1.0) # torch tensor
    s_ds = s_t.detach().cpu().numpy().astype(np.float32, copy=False) # (8000, 88)
    v = s_ds.ravel() # Flattened vector (8000 * 88).
    return v, s_ds
    
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
            v, _ = preprocess_feature_array(s, n_mels=n_mels, drop_mels=drop_mels)
            if v is None:
                continue
            batch.append(v[None, :])
            paths.append(fp)

        if len(batch) == batch_files:
            Xb = np.vstack(batch).astype(np.float32, copy=False)
            yield Xb, paths 
            batch, paths = [], []

    if batch:
        Xb = np.vstack(batch).astype(np.float32, copy=False)
        yield Xb, paths

def fit_streaming_pca_kmeans(data_path: Path, n_components: int = 256, k: int = 8, batch_files: int = 512
        , sample_size: int = 8000, random_state: int = 104, pcs_for_kmeans: int = 8):

    # Step 1: Partially fit the standard scaler on streaming batches.
    scaler = StandardScaler(with_mean=True, with_std=True)
    used_paths = []
    for Xb, paths in files_to_vectors_batch(data_path, batch_files=batch_files):
        scaler.partial_fit(Xb)
        used_paths.extend(paths)

    # Step 2: Fit IncrementalPCA on standardized batches.
    ipca = IncrementalPCA(n_components=n_components, batch_size=None)
    for Xb, _ in files_to_vectors_batch(data_path, batch_files=batch_files):
        Xb_std = scaler.transform(Xb)
        ipca.partial_fit(Xb_std)

    # Step 3: Train MiniBatchKMeans while transforming.
    rng = np.random.default_rng(random_state)
    keep = []  # Small reservoir of Z for silhouette score.
    kmeans = MiniBatchKMeans(
        n_clusters=k, init="k-means++", n_init="auto",
        batch_size=min(512, 256 * slurm_cpus()),
        max_iter=100, random_state=random_state
    )

    for Xb, _ in files_to_vectors_batch(data_path, batch_files=batch_files):
        Zb_full = ipca.transform(scaler.transform(Xb)).astype(np.float32, copy=False)
        Zb = Zb_full[:, :pcs_for_kmeans]
        kmeans.partial_fit(Zb)

        # Reserve sample for silhouette scores.
        if sample_size > 0:
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

def assign_labels_and_save(data_path: Path, scaler: StandardScaler, ipca: IncrementalPCA, kmeans: MiniBatchKMeans
    , used_paths: list[Path], batch_files: int = 512, key: str = "feature", n_mels: int = 128, drop_mels: int = 40
    , topn: int = 10, save_downsampled_dir: Path | None = None, pcs_for_kmeans: int = 8, output_path: Path | None = None):
    """
    Streaming pass that:
      - Computes labels and squared distance to centroid for each file.
      - Keeps top-N nearest per cluster,
      - Optionally writes per-file downsampled features to disk.
    Returns:
      labels: (N,) int32
      dist2: (N,) float32 squared distance to assigned centroid
      nearest: dict[cluster] -> list of tuples (dist2, path)
    """

    # Prepare output dir for downsampled features.
    if save_downsampled_dir is not None:
        save_downsampled_dir.mkdir(parents=True, exist_ok=True)

    k = kmeans.cluster_centers_.shape[0]
    nearest_heaps = [ [] for _ in range(k) ]  # max-heaps of (-dist2, path)
    labels = np.full(len(used_paths), -1, dtype=np.int32)
    dist2_assigned = np.full(len(used_paths), np.nan, dtype=np.float32)

    emb_list = [] # Collect 8D embeddings for saving.

    idx_iter = iter(range(len(used_paths)))
    file_iter = iter(used_paths)

    while True:
        batch = []
        bpaths = []
        bidxs = []

        try: 
            for _ in range(batch_files):
                j = next(idx_iter)
                p = next(file_iter)

                with np.load(p, mmap_mode="r") as z:
                    if key not in z.files:
                        continue 
                    s = np.squeeze(z[key])
                    v, s_ds = preprocess_feature_array(s, n_mels=n_mels, drop_mels=drop_mels)
                    if v is None:
                        continue
                    batch.append(v)
                    bpaths.append(p)
                    bidxs.append(j)

                    # Save the downsampled feature if true.
                    if save_downsampled_dir is not None:
                        out_fp = save_downsampled_dir / (p.stem + ".npz")
                        # Store downsampled spectrogram s_ds as 'feature_ds' to distinguish from original.
                        np.savez_compressed(out_fp, feature_ds=s_ds)
        except StopIteration:
            pass

        if not batch:
            break

        Xb = np.vstack(batch).astype(np.float32, copy=False)
        Zb_full = ipca.transform(scaler.transform(Xb)).astype(np.float32, copy=False)
        Zb = Zb_full[:, :pcs_for_kmeans]

        # Assign clusters and distances.
        # dist^2 = ||z||^2 + ||c||^2 - 2 z.c.
        x2 = np.sum(Zb**2, axis=1, keepdims=True)
        c2 = np.sum(kmeans.cluster_centers_**2, axis=1, keepdims=True).T
        xc = Zb @ kmeans.cluster_centers_.T
        d2_all = x2 + c2 - 2.0 * xc   # (B, k)

        labs = np.argmin(d2_all, axis=1).astype(np.int32)
        d2_assigned_b = d2_all[np.arange(len(labs)), labs].astype(np.float32)

        emb_list.append(Zb)

        # Record outputs into full arrays.
        for lab, pth, d2, orig_idx in zip(labs, bpaths, d2_assigned_b, bidxs):
            labels[orig_idx] = lab
            dist2_assigned[orig_idx] = d2
            h = nearest_heaps[lab]
            item = (-float(d2), str(pth))
            if len(h) < topn:
                heapq.heappush(h, item)
            else: 
                if -h[0][0] > float(d2):
                    heapq.heapreplace(h, item)
            
    Z8 = np.vstack(emb_list).astype(np.float32, copy=False) if emb_list else None

    Z8_2d = None
    if Z8 is not None and len(Z8) >= 2:
        pca2 = PCA(n_components=2, random_state=0)
        Z8_2d = pca2.fit_transform(Z8).astype(np.float32, copy=False)

    # Save embeddings if output_path provided
    if output_path is not None:
        if Z8 is not None:
            np.save(output_path / "embedding_pca8.npy", Z8)
        if Z8_2d is not None:
            np.save(output_path / "embedding_pca8_2d.npy", Z8_2d)

    # Turn heaps into sorted nearest lists.
    nearest = {}
    for j in range(k):
        items = sorted([(-neg_d2, Path(p)) for (neg_d2, p) in nearest_heaps[j]], key=lambda x: x[0])
        nearest[j] = items
    return labels, dist2_assigned, nearest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=str, required=True, help="Folder containing .npz feature files.")
    ap.add_argument("--output-root", type=str, required=True, help="Folder to write outputs.")
    ap.add_argument("--n-mels", type=int, default=128, help="Feature dimension expected per frame.")
    ap.add_argument("--var-thresh", type=float, default=0.95, help="Cumulative variance threshold for PCA.")
    ap.add_argument("--clusters", type=int, default=8, help="Number of KMeans clusters.")
    # Variables for downsampling and mapping.
    ap.add_argument("--save-assignments", action="store_true", help="Whether to save per-file downsampled features and assignments.")
    ap.add_argument("--topn", type=int, default=10, help="Number of nearest samples to keep per cluster.")
    ap.add_argument("--save-downsampled-dir", type=str, default="", help="Folder to save downsampled features. If empty, do not save.")
    ap.add_argument("--pcs-for-kmeans", type=int, default=8, help="Number of PCA components to use for k-means clustering.")
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
        , k=K, batch_files=N_BATCH_FILES, sample_size=8_000, random_state=104, pcs_for_kmeans=args.pcs_for_kmeans)

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
    if args.save_assignments or args.save_downsampled_dir:
        ds_dir = Path(args.save_downsampled_dir) if args.save_downsampled_dir else None
        labels, dist2, nearest = assign_labels_and_save(
            data_path=data_path,
            scaler=scaler, ipca=ipca, kmeans=kmeans, used_paths=used_paths,
            batch_files=N_BATCH_FILES, key="feature",
            n_mels=args.n_mels, drop_mels=40,
            topn=args.topn, save_downsampled_dir=ds_dir,
            pcs_for_kmeans=args.pcs_for_kmeans, output_path=output_path
        )

        # Persist aligned arrays
        np.save(output_path / "cluster_labels.npy", labels)
        np.save(output_path / "dist2_to_centroid.npy", dist2)

        # Write a CSV of the nearest items per cluster
        csv_path = output_path / "nearest_to_centroids.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("cluster,rank,dist2,feature_path\n")
            for j, items in nearest.items():
                for r, (d2, pth) in enumerate(items, start=1):
                    f.write(f"{j},{r},{d2:.6f},{pth}\n")

        print(f"[done] Saved assignments to {output_path}")
        if ds_dir is not None:
            print(f"[done] Downsampled features written to: {ds_dir}")

if __name__ == "__main__":
    main()