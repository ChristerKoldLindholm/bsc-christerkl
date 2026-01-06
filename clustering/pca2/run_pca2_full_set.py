#!/usr/bin/env python
# (1) Loads the full feature set from a Slurm folder.
# (2) Segments them into 5s log-mel segments and flattens.
# (3) Normalizes features (StandardScaler).
# (4) Performs IncrementalPCA on the full data set.
# (5) Runs k-means for a single chosen (n_components, k).
# (6) Also computes a 2D PCA projection for plotting.
# (7) Saves all results for later reference.

import argparse
from collections import defaultdict
import heapq
import multiprocessing
import numpy as np 
from numpy.lib.format import open_memmap
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import sys

import_path = Path.cwd().parents[1]
os.sys.path.insert(0, str(import_path))
import feature_utils as futils

def slurm_cpus():
    try:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", "").strip() or 0) or multiprocessing.cpu_count()
    except Exception:
        return multiprocessing.cpu_count()
    
def list_feature_files(folder: Path):
    files = sorted(folder.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")
    return files 
    
def stream_segments(files, batch_segments:int, segment_sec:float=5.0
                    , clip_sec:float=270.0
                    , mels_start:int=9, mels_end:int=60
                    , downsample_T:int=300, key:str="feature"):
    """
    Yield (X_batch, meta_batch) where X_batch is (B, D) matrix of flattened segment features.
    meta_batch: list of dicts with keys: file_path, segment_idx, start_frame, end_frame.
    """
    batch = []
    meta_batch = []

    for fp in files:
        with np.load(fp, mmap_mode="r") as z:
            if key not in z.files:
                continue
            s = np.squeeze(z[key])
            s = s[mels_start:mels_end, :] # Drop lowest mel bins.
            if s.ndim != 2:
                continue
            M, T = s.shape

            frames_per_segment = int(round(T*(segment_sec/clip_sec)))
            if frames_per_segment <= 0 or frames_per_segment > T:
                continue

            seg_idx = 0
            for start in range(0, T - frames_per_segment + 1, frames_per_segment):
                end = start + frames_per_segment
                seg = s[:, start:end]
                seg = futils.downsample_time_avgpool_from_db(seg, T_target=downsample_T, ref=1.0).numpy()
                v = seg.astype(np.float32, copy=False).ravel()[None, :] # (1, D).

                batch.append(v)
                meta_batch.append(dict(
                    file_path=str(fp),
                    segment_idx=seg_idx,
                    start_frame=int(start),
                    end_frame=int(end)
                ))
                seg_idx += 1
                if len(batch) == batch_segments:
                    Xb = np.vstack(batch).astype(np.float32, copy=False)
                    yield Xb, meta_batch
                    batch, meta_batch = [], []
    if batch:
        Xb = np.vstack(batch).astype(np.float32, copy=False)
        yield Xb, meta_batch
    
def fit_kmeans_streaming(files, transform_fn, n_pc:int, k:int, batch_segments:int
                         , segment_sec:float, clip_sec:float, mels_start:int, mels_end:int
                         , downsample_T:int, random_state:int=104):
    """
    One streaming pass to partial_fit MiniBatchKMeans on Z[:, :n_pc].
    Keep a reservoir sample for silhouette estimates.
    """
    rng = np.random.default_rng(random_state)
    keep = []

    kmeans = MiniBatchKMeans(
        n_clusters=k, init="k-means++", n_init="auto", batch_size=min(8192, 256 * slurm_cpus()),
        max_iter=100, random_state=random_state)
    total_seen = 0

    for Xb, _ in stream_segments(files, batch_segments=batch_segments, segment_sec=segment_sec
                                 , mels_start=mels_start, mels_end=mels_end
                                 , downsample_T=downsample_T, clip_sec=clip_sec):
        Zb = transform_fn(Xb, n_pc)
        kmeans.partial_fit(Zb)
        keep.append(Zb)
        total_seen += len(Zb)

        if total_seen > 54_000: # Cap memory at n segments. 
            Ztmp = np.vstack(keep)
            idx = rng.choice(len(Ztmp), min(32_000, len(Ztmp)), replace=False)
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

def assign_labels_and_distances(files, transform_fn, n_pc:int, kmeans: MiniBatchKMeans
                                , segment_meta_out:Path, labels_out:Path
                                , dist2_out:Path, nearest_csv_out:Path
                                , topn:int, batch_segments:int
                                , segment_sec:float, clip_sec:float
                                , mels_start:int=9, mels_end:int=60, downsample_T:int=300):
    """
    Second streaming pass: predict labels and squared distances using
       d^2 = ||z||^2 + ||c||^2 - 2 z.c
    Save:
        - labels_out: cluster label per segment.
        - dist2_out: squared distance to assigned centroid.
        - segment_meta_out: metadata with segment mapping.
        - nearest_csv_out: CSV with top-N nearest segments per cluster.
    """
    
    meta_all = []
    k = kmeans.cluster_centers_.shape[0]
    centers = kmeans.cluster_centers_.astype(np.float32, copy=False)
    c2 = np.sum(centers**2, axis=1, keepdims=True).T  # (1, k)

    labels_all = []
    dist2_all = []
    nearest_heaps = [[] for _ in range(k)]  # max-heaps of (-dist2, meta)

    for Xb, meta_batch in stream_segments(files
                                          , batch_segments=batch_segments
                                          , segment_sec=segment_sec
                                          , clip_sec=clip_sec
                                          , mels_start=mels_start
                                          , mels_end=mels_end
                                          , downsample_T=downsample_T):
        Zb = transform_fn(Xb, n_pc) # (B, n_pc)
        x2 = np.sum(Zb**2, axis=1, keepdims=True) # (B, 1)
        xc = Zb @ centers.T # (B, k)
        d2_all_block = x2 + c2 - 2.0 * xc # (B, k)

        labs = np.argmin(d2_all_block, axis=1).astype(np.int32)
        d2_assigned = d2_all_block[np.arange(len(labs)), labs].astype(np.float32, copy=False)
        labels_all.append(labs)
        dist2_all.append(d2_assigned)
        meta_all.extend(meta_batch)

        # Keep path and segment_idx for nearest segments per cluster.
        for j, meta, d2 in zip(labs, meta_batch, d2_assigned):
            h = nearest_heaps[j]
            desc = f'{meta["file_path"]}#seg{meta["segment_idx"]}'
            item = (-float(d2), desc)
            if len(h) < topn:
                heapq.heappush(h, item)
            else:
                if -h[0][0] > float(d2):
                    heapq.heapreplace(h, item)

    labels_all = np.concatenate(labels_all) if labels_all else np.empty((0,), dtype=np.int32)
    dist2_all = np.concatenate(dist2_all) if dist2_all else np.empty((0,), dtype=np.float32)

    np.save(labels_out, labels_all)
    np.save(dist2_out, dist2_all)

    # Segment meta mapping.
    meta_dtype = np.dtype(
        [
            ("file_path", "U256")
            , ("segment_idx", "i4")
            , ("start_frame", "i4")
            , ("end_frame", "i4")
        ]
    )

    meta_array = np.empty(len(meta_all), dtype=meta_dtype)
    for i, m in enumerate(meta_all):
        meta_array[i]["file_path"] = m["file_path"]
        meta_array[i]["segment_idx"] = m["segment_idx"]
        meta_array[i]["start_frame"] = m["start_frame"]
        meta_array[i]["end_frame"] = m["end_frame"]
    np.save(segment_meta_out, meta_array)

    # Write nearest per cluster.
    with open(nearest_csv_out, "w", encoding="utf-8") as f:
        f.write("cluster,rank,dist2,feature_segment\n")
        for j in range(k):
            items = sorted([(-neg_d2, desc) for (neg_d2, desc) in nearest_heaps[j]]
                           , key=lambda x: x[0])
            for r, (d2, desc) in enumerate(items, start=1):
                f.write(f"{j},{r},{d2:.6f},{desc}\n")    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=str, required=True, help="Folder containing .npz log-mel features.")
    ap.add_argument("--output-root", type=str, required=True, help="Folder to write outputs.")
    ap.add_argument("--batch-segments", type=int, default=1024, help="Segments per streaming batch.")
    ap.add_argument("--segment-sec", type=float, default=5.0, help="Seconds per segment.")
    ap.add_argument("--clip-sec", type=float, default=270.0, help="Expected length of full clip in seconds.")
    ap.add_argument("--topn", type=int, default=500, help="Top-N nearest segments to save per cluster.")
    ap.add_argument("--n-components", type=int, required=True, help="Number of principal components used for k-means.")
    ap.add_argument("--k", type=int, required=True, help="Number of clusters for k-means.")
    ap.add_argument("--n-mels", type=int, default=128, help="Number of mel bins in input features.")
    ap.add_argument("--mels-start", type=int, default=9, help="Start index of mel bins to use.")
    ap.add_argument("--mels-end", type=int, default=60, help="End index of mel bins to use.")
    ap.add_argument("--downsample-target", type=int, default=300, help="Target downsample rate for features.")
    ap.add_argument("--random-seed", type=int, default=104, help="RNG seed for k-means and sampling inside silhouette.")
    args = ap.parse_args()
    ds_target = args.downsample_target

    data_path = Path(args.input_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = list_feature_files(data_path)
    np.save(out_root / "all_paths.npy", np.array([str(f) for f in files], dtype="U4096"))

    max_pc = args.n_components

    # (1) Fit IncrementalPCA once on all 5-sec segments.
    scaler = StandardScaler(with_mean=True, with_std=True)
    for Xb, _ in stream_segments(files
                                 , batch_segments=args.batch_segments
                                 , segment_sec=args.segment_sec
                                 , clip_sec=args.clip_sec
                                 , mels_start=args.mels_start
                                 , mels_end=args.mels_end
                                 , downsample_T=ds_target):
        scaler.partial_fit(Xb)
    
    scaler_mean = scaler.mean_.astype(np.float32, copy=False)
    scaler_scale = scaler.scale_.astype(np.float32, copy=False)
    scaler_scale[scaler_scale == 0.0] = 1.0  # 0-division safeguard.

    # (2) Fit IncrementalPCA on normalized data.
    ipca = IncrementalPCA(n_components=max_pc)
    
    for Xb, _ in stream_segments(files
                                 , batch_segments=args.batch_segments
                                 , segment_sec=args.segment_sec
                                 , clip_sec=args.clip_sec
                                 , mels_start=args.mels_start
                                 , mels_end=args.mels_end
                                 , downsample_T=ds_target):
        Xb_sc = (Xb - scaler_mean) / scaler_scale
        ipca.partial_fit(Xb_sc)
    
    ipca_components = ipca.components_.astype(np.float32, copy=False)
    ipca_mean = ipca.mean_.astype(np.float32, copy=False)

    np.savez_compressed(
        out_root / "segments_ipca.npz",
        ipca_components=ipca_components,
        ipca_mean=ipca_mean,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        explained_variance=ipca.explained_variance_.astype(np.float32, copy=False),
        explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32, copy=False),
        n_components=int(max_pc),
    )

    # Save configs.
    np.savez_compressed(
        out_root / "config_and_ipca.npz",
        ipca_components=ipca_components,
        ipca_mean=ipca_mean,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        explained_variance=ipca.explained_variance_.astype(np.float32, copy=False),
        explained_variance_ratio=ipca.explained_variance_ratio_.astype(
            np.float32, copy=False
        ),
        segment_sec=float(args.segment_sec),
        clip_sec=float(args.clip_sec),
        mels_start=int(args.mels_start),
        mels_end=int(args.mels_end),
        downsample_target=int(args.downsample_target),
        n_mels=int(args.n_mels),
        n_components=int(max_pc),
    )

    def T_any(Xb, n_pc):
        # Xb: (B, D)
        Xb_sc = (Xb - scaler_mean) / scaler_scale
        # Both mean centering and standard deviation scaling.
        Z = (Xb_sc - ipca_mean) @ ipca_components[:n_pc].T
        return Z.astype(np.float32, copy=False)
    
    pc_dir = out_root / f"pc_{args.n_components:03d}"
    pc_dir.mkdir(parents=True, exist_ok=True)
    kdir = pc_dir / f"k_{args.k:02d}"
    kdir.mkdir(parents=True, exist_ok=True)
    
    # (3) Fit k-means on the full dataset in the chosen PC space.
    kmeans, sil = fit_kmeans_streaming(files=files
                                       , transform_fn=T_any
                                       , n_pc=args.n_components
                                       , k=args.k
                                       , batch_segments=args.batch_segments
                                       , segment_sec=args.segment_sec
                                       , clip_sec=args.clip_sec
                                       , mels_start=args.mels_start
                                       , mels_end=args.mels_end
                                       , downsample_T=ds_target
                                       , random_state=args.random_seed)
    
    np.save(kdir / "kmeans_centers.npy", kmeans.cluster_centers_.astype(np.float32, copy=False))
    np.savez_compressed(
        kdir / "kmeans_summary.npz",
        k=np.int32(args.k),
        inertia=np.float32(kmeans.inertia_),
        silhouette=np.float32(sil),
        n_pc=np.int32(args.n_components)
    )

    # (4) Assign labels and distances for all segments.
    assign_labels_and_distances(files=files
                                , transform_fn=T_any
                                , n_pc=args.n_components
                                , kmeans=kmeans
                                , segment_meta_out=kdir / "segment_meta.npy"
                                , labels_out=kdir / "segment_labels.npy"
                                , dist2_out=kdir / "segment_dist2.npy"
                                , nearest_csv_out=kdir / "nearest_segments.csv"
                                , topn=args.topn
                                , batch_segments=args.batch_segments
                                , segment_sec=args.segment_sec
                                , clip_sec=args.clip_sec
                                , mels_start=args.mels_start
                                , mels_end=args.mels_end
                                , downsample_T=ds_target
                                )
    
    # (5) Compute and save PCA scores: Z_opt: chosen n_components from elbow experiment. 
    # Z_2: PCA(2) for 2D plotting.
    print("[info] Counting total number of segments for PCA scores...")
    total_segments = 0 
    for _, meta_batch in stream_segments(files
                                        , batch_segments=args.batch_segments
                                        , segment_sec=args.segment_sec
                                        , clip_sec=args.clip_sec
                                        , mels_start=args.mels_start
                                        , mels_end=args.mels_end
                                        , downsample_T=ds_target):
        total_segments += len(meta_batch)
    print(f"[info] Total segments: {total_segments}")

    if total_segments == 0:
        print("[info] No segments found, skipping PCA score computation.")
        np.save(pc_dir / "pca_scores.npy", np.empty((0, args.n_components), dtype=np.float32))
        np.save(pc_dir / "pca2_scores.npy", np.empty((0, 2), dtype=np.float32))
        return
    
    pca_scores_path = pc_dir / "pca_scores.npy"
    pca2_scores_path = pc_dir / "pca2_scores.npy"

    Z_opt_mm = open_memmap(
        pca_scores_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_segments, args.n_components),
    )

    Z2_mm = open_memmap(
        pca2_scores_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_segments, 2),
    )
    offset = 0

    for Xb, meta_batch in stream_segments(files
                                , batch_segments=args.batch_segments
                                , segment_sec=args.segment_sec
                                , clip_sec=args.clip_sec
                                , mels_start=args.mels_start
                                , mels_end=args.mels_end
                                , downsample_T=ds_target
                                ):
        B = len(meta_batch)
        if B == 0:
            continue
        Z_opt = T_any(Xb, args.n_components)
        Z2 = T_any(Xb, 2)
        Z_opt_mm[offset : offset + B, :] = Z_opt
        Z2_mm[offset : offset + B, :] = Z2
        offset += B

    del Z_opt_mm
    del Z2_mm
    
    print("[info] PCA scores computation completed.")

if __name__ == "__main__":
    main()