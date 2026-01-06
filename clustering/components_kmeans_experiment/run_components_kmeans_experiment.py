#!/usr/bin/env python3
"""
Run k-means on saved IncrementalPCA (256 components) for multiple PC counts, with elbow (k=2..16).
Saves: per-PC elbow arrays, and per-(PC,k) labels, dist2, centers, and nearest-to-centroid CSV.

Run as Slurm batch job:
  python run_kmeans_sweep_saved_pca.py \
    --input-root /path/to/features_root \
    --saved-pca /path/to/pca_kmeans_batches.npz \
    --output-root /path/to/out_sweep \
    --pc-list 2 3 4 5 6 7 8 16 32 64 128 256 \
    --batch-files 512 --n-mels 128 --drop-mels 40 --topn 10
"""

import argparse
import heapq
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import os
import sys
import multiprocessing

utils_path = Path.cwd().parents[1]
sys.path.insert(0, str(utils_path))
import feature_utils as futils # Downsampling function.

def slurm_cpus():
    try:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", "").strip() or 0) or multiprocessing.cpu_count()
    except Exception:
        return multiprocessing.cpu_count()
    
def list_downsampled_feature_files(folder: Path):
    files = sorted(folder.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")
    return files 

def vectors_from_saved_ds(files, batch_files:int=512, key="feature_ds"):
    batch, paths = [], []
    for fp in files:
        with np.load(fp, mmap_mode="r") as z:
            if key not in z.files:
                continue
            s_ds = np.squeeze(z[key])
            if s_ds.ndim != 2:
                continue 
            v = s_ds.astype(np.float32, copy=False).ravel()
            batch.append(v[None, :]); paths.append(str(fp))
        if len(batch) == batch_files:
            yield np.vstack(batch).astype(np.float32, copy=False), paths 
            batch, paths = [], []
    if batch:
        yield np.vstack(batch), paths 

def stream_vectors(files, use_saved_ds: bool, batch_files: int, n_mels: int, drop_mels: int):
    if use_saved_ds:
        yield from vectors_from_saved_ds(files, batch_files=batch_files, key="feature_ds")
        return 
    
    # Fallback: recompute downsampled features.
    batch, paths = [], []
    for fp in files:
        with np.load(fp, mmap_mode="r") as z:
            if "feature" not in z.files:
                continue
            s = np.squeeze(z["feature"])
            v, _ = futils.preprocess_feature_array(s, n_mels=n_mels, drop_mels=drop_mels)
            if v is None:
                continue
            batch.append(v[None, :]); paths.append(str(fp))
            if len(batch) == batch_files:
                yield np.vstack(batch).astype(np.float32, copy=False), paths
                batch, paths = [], []
    if batch:
        yield np.vstack(batch).astype(np.float32, copy=False), paths

def project_with_saved_pca(Xb, scaler_mean, scaler_scale, ipca_mean, ipca_components, n_pc: int):
    """
    Applies saved StandardScaler + IncrementalPCA transform:
        X_std = (X - scaler_mean) / scaler_scale
        Z = (X_std - ipca_mean) @ ipca_components.T
    Then returns Z[:, :n_pc] as float32.
    """
    X_std = (Xb - scaler_mean) / scaler_scale
    Z = (X_std - ipca_mean) @ ipca_components.T
    return Z[:, :n_pc].astype(np.float32, copy=False)

def fit_kmeans_streaming(files, transform_fn, n_pc: int, k: int, batch_files: int
                         , n_mels: int, drop_mels: int, use_saved_ds: bool, random_state: int = 104):
    """
    One streaming pass to partial_fit MiniBatchKMeans on Z[:, :n_pc].
    Also keep a reservoir sample for silhouette estimates.
    """
    rng = np.random.default_rng(random_state)
    keep = []
    kmeans = MiniBatchKMeans(
        n_clusters=k, init="k-means++", n_init="auto",
        batch_size=min(8192, 256 * slurm_cpus()),
        max_iter=100, random_state=random_state
    )
    total_seen = 0
    for Xb, _ in stream_vectors(files, use_saved_ds=use_saved_ds, batch_files=batch_files
            , n_mels=n_mels, drop_mels=drop_mels):
        Zb = transform_fn(Xb, n_pc)
        kmeans.partial_fit(Zb)
        # Keep reservoir for silhouette.
        keep.append(Zb)
        total_seen += len(Zb)
        if total_seen > 16000:  # Cap memory.
            Ztmp = np.vstack(keep)
            idx = rng.choice(len(Ztmp), min(8000, len(Ztmp)), replace=False)
            keep = [Ztmp[idx]]
            total_seen = len(keep[0])
    sil = np.nan
    if keep:
        Zs = np.vstack(keep)
        if len(Zs) > k:
            labs_s = kmeans.predict(Zs)
            if len(np.unique(labs_s)) > 1:
                sil = float(silhouette_score(Zs, labs_s))
    return kmeans, sil

def assign_labels_and_distances(files, transform_fn, n_pc: int, kmeans: MiniBatchKMeans,
                                used_paths_out: Path, labels_out: Path, dist2_out: Path,
                                nearest_csv_out: Path, topn: int, batch_files: int,
                                n_mels: int, drop_mels: int, use_saved_ds: bool):
    """
    Second streaming pass: predict labels and squared distances using
       d^2 = ||z||^2 + ||c||^2 - 2 z.c
    Keep top-N nearest file paths per cluster and save labels + dist² aligned with used_paths.
    """
    # We also persist the ordered used_paths we iterated (to align with labels/dist2)
    used_paths = []

    k = kmeans.cluster_centers_.shape[0]
    centers = kmeans.cluster_centers_.astype(np.float32, copy=False)
    c2 = np.sum(centers**2, axis=1, keepdims=True).T  # (1, k)

    labels_all = []
    dist2_all = []
    nearest_heaps = [[] for _ in range(k)]  # max-heaps of (-dist2, path)

    for Xb, paths in stream_vectors(files, use_saved_ds=use_saved_ds, batch_files=batch_files
            , n_mels=n_mels, drop_mels=drop_mels):
        Zb = transform_fn(Xb, n_pc)  # (B, n_pc)
        x2 = np.sum(Zb**2, axis=1, keepdims=True) # (B, 1)
        xc = Zb @ centers.T # (B, k)
        d2_all_block = x2 + c2 - 2.0 * xc # (B, k)
        labs = np.argmin(d2_all_block, axis=1).astype(np.int32)
        d2_assigned = d2_all_block[np.arange(len(labs)), labs].astype(np.float32, copy=False)

        labels_all.append(labs)
        dist2_all.append(d2_assigned)
        used_paths.extend([str(p) for p in paths])

        for j, pth, d2 in zip(labs, paths, d2_assigned):
            h = nearest_heaps[j]
            item = (-float(d2), str(pth))
            if len(h) < topn:
                heapq.heappush(h, item)
            else:
                if -h[0][0] > float(d2):
                    heapq.heapreplace(h, item)

    labels_all = np.concatenate(labels_all) if labels_all else np.empty((0,), dtype=np.int32)
    dist2_all = np.concatenate(dist2_all) if dist2_all else np.empty((0,), dtype=np.float32)
    np.save(used_paths_out, np.array(used_paths))
    np.save(labels_out, labels_all)
    np.save(dist2_out, dist2_all)

    # Write nearest per cluster
    with open(nearest_csv_out, "w", encoding="utf-8") as f:
        f.write("cluster,rank,dist2,feature_path\n")
        for j in range(k):
            items = sorted([(-neg_d2, p) for (neg_d2, p) in nearest_heaps[j]], key=lambda x: x[0])
            for r, (d2, pth) in enumerate(items, start=1):
                f.write(f"{j},{r},{d2:.6f},{pth}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=str, required=True, help="Folder containing .npz feature files.")
    ap.add_argument("--saved-pca", type=str, required=True, help="Path to prior pca_kmeans_batches.npz with scaler+PCA.")
    ap.add_argument("--output-root", type=str, required=True, help="Folder to write outputs.")
    ap.add_argument("--pc-list", type=int, nargs="+", default=[2,3,4,5,6,7,8,16,32,64,128,256], help="PC counts to sweep.")
    ap.add_argument("--batch-files", type=int, default=512)
    ap.add_argument("--n-mels", type=int, default=128)
    ap.add_argument("--drop-mels", type=int, default=40)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--downsampled-dir", type=str, default=None, help="If set, use these pre-downsampled features.")
    ap.add_argument("--save-plotting", action="store_true", help="Save Z2 and per(PC,k) centroid_2d.")
    args = ap.parse_args()

    data_path = Path(args.input_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load saved PCA and scaler.
    saved = np.load(args.saved_pca, allow_pickle=False)
    scaler_mean = saved["scaler_mean"]
    scaler_scale = saved["scaler_scale"]
    ipca_components = saved["ipca_components"]
    ipca_mean = saved["ipca_mean"]

    if args.downsampled_dir:
        files = list_downsampled_feature_files(Path(args.downsampled_dir))
        use_saved_ds = True
    else:
        files = sorted(Path(args.input_root.rglob("*.npz")))
        use_saved_ds = False 

    def T_any(Xb, n_pc):
        X_std = (Xb - scaler_mean) / scaler_scale
        Z = (X_std - ipca_mean) @ ipca_components.T
        return Z[:, :n_pc].astype(np.float32, copy=False)
    use_saved_ds = True 
    if args.save_plotting:
        out_plot = Path(args.output_root) / "plotting"
        out_plot.mkdir(parents=True, exist_ok=True)
        used_paths_all, Z2_all = [], []
        for Xb, paths in stream_vectors(files, use_saved_ds=use_saved_ds, batch_files=args.batch_files, n_mels=args.n_mels, drop_mels=args.drop_mels):
            Z2 = T_any(Xb, 2)
            Z2_all.append(Z2)
            used_paths_all.extend(paths)
        if Z2_all:
            Z2_all = np.vstack(Z2_all)
            np.save(out_plot / "used_paths.npy", np.array(used_paths_all, dtype=object))
            np.save(out_plot / "Z2.npy", Z2_all.astype(np.float32, copy=False))


    ks = list(range(2, 17))  # Elbow range.

    # Elbow experiment.
    for n_pc in args.pc_list:
        pc_dir = out_root / f"pc_{n_pc:03d}"
        pc_dir.mkdir(parents=True, exist_ok=True)

        inertias = []
        silhouettes = []
        # Run elbow (k=2..16).
        for k in ks:
            km, sil = fit_kmeans_streaming(
                files=files, transform_fn=T_any, n_pc=n_pc, k=k,
                batch_files=args.batch_files, n_mels=args.n_mels, drop_mels=args.drop_mels, 
                use_saved_ds=use_saved_ds, random_state=104
            )
            inertias.append(float(km.inertia_))
            silhouettes.append(float(sil) if np.isfinite(sil) else np.nan)

            # Save per-(PC,k) results
            kdir = pc_dir / f"k_{k:02d}"
            kdir.mkdir(parents=True, exist_ok=True)
            np.save(kdir / "kmeans_centers.npy", km.cluster_centers_.astype(np.float32, copy=False))

            if args.save_plotting:
                np.save(kdir / "kmeans_centers_2d.npy", km.cluster_centers_[:, :2].astype(np.float32, copy=False))

            # Assign labels + distances and write nearest mapping
            assign_labels_and_distances(
                files=files, transform_fn=T_any, n_pc=n_pc, kmeans=km,
                used_paths_out=kdir / "used_paths.npy",
                labels_out=kdir / "cluster_labels.npy",
                dist2_out=kdir / "dist2_to_centroid.npy",
                nearest_csv_out=kdir / "nearest_to_centroids.csv",
                topn=args.topn, batch_files=args.batch_files,
                n_mels=args.n_mels, drop_mels=args.drop_mels, use_saved_ds=use_saved_ds
            )

        # Save elbow for this PC count
        np.savez_compressed(
            pc_dir / "elbow.npz",
            k_values=np.array(ks, dtype=np.int32),
            inertias=np.array(inertias, dtype=np.float64),
            silhouettes=np.array(silhouettes, dtype=np.float64),
            n_pc=int(n_pc)
        )

if __name__ == "__main__":
    main()