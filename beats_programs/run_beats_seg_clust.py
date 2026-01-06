#!/usr/bin/env python3

import argparse
import random 
import heapq
import numpy as np
from pathlib import Path

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import torch
import torchaudio

import os
import multiprocessing
import sys 

# Segment meta mapping.
meta_dtype = np.dtype(
    [
        ("file_path", "U256")
        , ("segment_idx", "i4")
        , ("start_frame", "i4")
        , ("end_frame", "i4")
    ]
)

def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def slurm_cpus():
    try:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", "").strip() or 0) or multiprocessing.cpu_count()
    except Exception:
        return multiprocessing.cpu_count()
    
def list_audio_files(root: Path, exts=(".wav")):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    files.sort()
    return files 

def load_audio_16k(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav

def iter_segments(wav: torch.Tensor, sr: int, segment_sec: float
                  , hop_sec: float, clip_sec: int):
    """
    Yields (segment_1d, start_sec, end_sec) where segment_1d = (t_seg,)."""
    assert wav.ndim() == 2 and wav.size(0) == 1
    T = wav.size(1)
    clip_T = min(T, int(round(clip_sec * sr)))
    seg_len = int(round(segment_sec * sr))
    hop_len = int(round(hop_sec * sr))
    if seg_len <= 0 or hop_len <= 0:
        raise ValueError("segment_sec and hop_sec must be >0.")
    
    seg_idx = 0
    start = 0    
    while start + seg_len <= clip_T:
        end = start + seg_len
        yield seg_idx, start, end, wav[:, start:end]
        seg_idx += 1
        start += hop_len 

def collect_audio_files(input_root: Path, sample_files: int, seed: int):
    exts = (".wav")
    files = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    if sample_files is not None and sample_files > 0 and sample_files < len(files):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(files), size=sample_files, replace=False)
        files = [files[i] for i in idx]
        files.sort()
    return files

def load_beats_model(device: torch.device, ckpt_path: Path):
    beats_path = Path.cwd().parents[1] / "unilm" / "beats" 
    sys.path.append(beats_path)
    from BEATs import BEATs, BEATsConfig

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    return model 

@torch.no_grad()
def beats_embed_batch(model, wav_batch, device: torch.device):
    """
    wav_batch: (N, t) mono 16 kHz waveform segments where t is 5 sec @ 16 kHz.
    returns: numpy (B, D) embeddings.
    """
    x = torch.cat(wav_batch, dim=0).to(device)
    if x.ndim == 2:
        pass 
    else:
        raise RuntimeError(f"Unexpected wav_batch ndim: {x.ndim}")

    feat, padding_mask = model.extract_features(x, padding_mask=None)

    return feat.detach().cpu().numpy().astype(np.float32, copy=False)

def extract_and_save_features(files, out_dir: Path, beats_ckpt: str
                              , segment_sec: float, clip_sec: float
                              , hop_sec: float, batch_segments: int
                              , device: torch.device, overwrite: bool = False):
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    config_path = feat_dir / "beats_config.json"
    if config_path.exists() and not overwrite:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config

    model = load_beats_model(device, beats_ckpt)
    batch_feats = []
    batch_meta = []
    batch_id = 0
    total_segments = 0
    emb_dim = None 

    def flush():
        nonlocal batch_id, total_segments, emb_dim
        if not batch_feats:
            return 
        F = np.concatenate(batch_feats, axis=0) # (N, D).
        M = np.array(batch_meta, dtype=meta_dtype) # (N,).
        if emb_dim is None:
            emb_dim = int(F.shape[1])
        
        np.save(feat_dir / f"features_batch{batch_id:04d}.npy", F)
        np.save(feat_dir / f"meta_batch{batch_id:04d}.npy", M)
        
        total_segments += F.shape[0]
        batch_id += 1
        batch_feats.clear()
        batch_meta.clear()

    sr = 16000
    wavbuf = []
    metabuf = []

    for fp in files:
        wav = load_audio_16k(str(fp), target_sr=sr) # (1, T)

        for seg_idx, s, e, seg in iter_segments(wav, sr, segment_sec, hop_sec, clip_sec):
            wavbuf.append(seg) # (1, t_seg).
            metabuf.append(str(fp), int(seg_idx), int(s), int(e))

            if len(wavbuf) >= batch_segments:
                feats = beats_embed_batch(model, wavbuf, device=device) # (B, D).
                batch_feats.append(feats)
                batch_meta.extend(metabuf)
                wavbuf.clear()
                metabuf.clear()
                flush()
    
    if wavbuf: # Any leftovers.
        feats = beats_embed_batch(model, wavbuf, device=device) # (B, D).
        batch_feats.append(feats)
        batch_meta.extend(metabuf)
        wavbuf.clear()
        metabuf.clear()
        flush()

    config = {
        "feature_dir": str(feat_dir),
        "num_batches": batch_id,
        "total_segments": total_segments,
        "embedding_dim": emb_dim,
        "sr": sr,
        "segment_sec": segment_sec,
        "hop_sec": hop_sec,
        "clip_sec": clip_sec,
        "meta_dtype": "file_path(U256), segment_idx(i4), start_frame(i4), end_frame(i4)",
        "beats_ckpt": beats_ckpt,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config

def iter_feature_batches(feature_dir: Path, num_batches: int):
    for i in range(num_batches):
        F = np.load(feature_dir / f"feats_{i:05d}.npy", mmap_mode="r")
        M = np.load(feature_dir / f"meta_{i:05d}.npy", mmap_mode="r")
        yield F, M

def run_pca_kmeans_elbow(files, transform_fn, n_pc:int, k:int, batch_segments:int
                         , segment_sec:float, clip_sec:float, mels_start:int, mels_end:int
                         , downsample_T:int=300, random_state:int=104):
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
                                 , clip_sec=clip_sec, downsample_T=downsample_T
                                 , key="feature"):
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
                                , segment_meta_out:Path
                                , labels_out:Path, dist2_out:Path, nearest_csv_out:Path
                                , topn:int, batch_segments:int
                                , segment_sec:float, clip_sec:float
                                , mels_start:int=9, mels_end:int=60
                                , downsample_T:int=300):
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
    ap.add_argument("--input-root", type=str, required=True, help="Folder containing .npz features.")
    ap.add_argument("--output-root", type=str, required=True, help="Folder to write outputs.")
    ap.add_argument("--max-pc", type=int, default=8192, help="Maximum number of principal components.")
    ap.add_argument("--min_i-pc", type=int, default=5, help="Minimum i where n_pc=2**i.")
    ap.add_argument("--pc_list", type=int, nargs="+", default=None, help="List of specific PC counts to use.")
    ap.add_argument("--batch-segments", type=int, default=1024, help="Segments per streaming batch.")
    ap.add_argument("--segment-sec", type=float, default=5.0, help="Seconds per segment.")
    ap.add_argument("--clip-sec", type=float, default=270.0, help="Expected length of full clip in seconds.")
    ap.add_argument("--topn", type=int, default=100, help="Top-N nearest segments to save per cluster.")
    ap.add_argument("--sample-files", type=int, default=2000, help="Only sample this many files for the experiment.")
    ap.add_argument("--sample-seed", type=int, default=104, help="RNG seed for sampling.")
    ap.add_argument("--n-mels", type=int, default=128)
    ap.add_argument("--mels-start", type=int, default=9)
    ap.add_argument("--mels-end", type=int, default=60)
    ap.add_argument("--downsample-target", type=int, default=300, help="Target time frames after downsampling.")
    ap.add_argument("--ks-list", type=int, nargs="+", default=None, help="List of k values to use.")
    args = ap.parse_args()
    ds_target = args.downsample_target
    # pc_list = [2**i for i in range(args.min_i_pc, args.max_pc) if 2**i <= args.max_pc]
    pc_list = args.pc_list
    ks = args.ks_list

    data_path = Path(args.input_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = list_feature_files(data_path)
    # Pick a sample subset of files for the elbow experiment.
    rng = np.random.default_rng(args.sample_seed)
    if args.sample_files and len(files) > args.sample_files:
        files = list(rng.choice(files, size=args.sample_files, replace=False))

    np.save(out_root / "sampled_paths.npy", np.array([str(f) for f in files], dtype="U4096"))

    # (1) Fit IncrementalPCA and normalize once on all 5-sec segments.
    scaler = StandardScaler(with_mean=True, with_std=True)

    for Xb, _ in stream_segments(
            files,
            batch_segments=args.batch_segments,
            segment_sec=args.segment_sec,
            clip_sec=args.clip_sec,
            mels_start=args.mels_start,
            mels_end=args.mels_end,
            downsample_T=args.downsample_target):
        scaler.partial_fit(Xb)

    max_pc = max(pc_list)
    ipca = IncrementalPCA(n_components=max_pc)
    
    for Xb, _ in stream_segments(files
                                 , batch_segments=args.batch_segments
                                 , segment_sec=args.segment_sec
                                 , clip_sec=args.clip_sec
                                 , mels_start=args.mels_start
                                 , mels_end=args.mels_end
                                 , downsample_T=ds_target
                                 , key="feature"):
        Xb_std = (Xb - scaler.mean_) / scaler.scale_
        ipca.partial_fit(Xb_std)
    
    ipca_components = ipca.components_.astype(np.float32, copy=False)
    ipca_mean = ipca.mean_.astype(np.float32, copy=False)

    np.savez_compressed(
        out_root / "segments_ipca.npz",
        ipca_components=ipca_components,
        ipca_mean=ipca_mean,
        scaler_mean=scaler.mean_.astype(np.float32, copy=False),
        scaler_scale=scaler.scale_.astype(np.float32, copy=False),
        explained_variance=ipca.explained_variance_.astype(np.float32, copy=False),
        explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32, copy=False)
    )

    scaler_mean = scaler.mean_.astype(np.float32, copy=False)
    scaler_scale = scaler.scale_.astype(np.float32, copy=False)

    def T_any(Xb, n_pc):
        # Xb: (B, D)
        Xb_std = (Xb - scaler_mean) / scaler_scale
        Z = (Xb_std - ipca_mean) @ ipca_components[:n_pc].T
        return Z.astype(np.float32, copy=False)
    
    # (2) For each PC count, fit KMeans and assign labels/distances.
    # ks = list(range(2, 17))
    print(f"Running segment clustering for PCs: {pc_list} and ks: {ks}")
    for n_pc in pc_list:
        print(f"Processing n_pc={n_pc}...")
        pc_dir = out_root / f"pc_{n_pc:03d}"
        pc_dir.mkdir(parents=True, exist_ok=True)

        inertias = []
        silhouettes = []

        for k in ks:
            print(f"  Fitting k={k}...")
            # Fit KMeans for this (PC, k).
            km, sil = fit_kmeans_streaming(files=files, transform_fn=T_any, n_pc=n_pc, k=k
                                           , batch_segments=args.batch_segments
                                           , segment_sec=args.segment_sec
                                           , clip_sec=args.clip_sec
                                           , mels_start=args.mels_start
                                           , mels_end=args.mels_end
                                           , downsample_T=ds_target
                                           , random_state=args.sample_seed)
            
            inertias.append(km.inertia_)
            silhouettes.append(sil)

            kdir = pc_dir / f"k_{k:02d}"
            kdir.mkdir(parents=True, exist_ok=True)

            np.save(kdir / "kmeans_centers.npy", km.cluster_centers_.astype(np.float32, copy=False))

            assign_labels_and_distances(
                files=files, transform_fn=T_any, n_pc=n_pc, kmeans=km,
                segment_meta_out=kdir / "segment_meta.npy",
                labels_out=kdir / "segment_labels.npy",
                dist2_out=kdir / "segment_dist2.npy",
                nearest_csv_out=kdir / "nearest_segments.csv",
                topn=args.topn,
                mels_start=args.mels_start,
                mels_end=args.mels_end,
                downsample_T=ds_target,
                batch_segments=args.batch_segments,
                segment_sec=args.segment_sec,
                clip_sec=args.clip_sec
            )

        np.savez_compressed(
            pc_dir / "kmeans_summary.npz",
            k_values=np.array(ks, dtype=np.int32),
            inertias=np.array(inertias, dtype=np.float32),
            silhouettes=np.array(silhouettes, dtype=np.float32),
            n_pc=int(n_pc)
        )

    np.savez_compressed(
    out_root / "config_and_ipca.npz",
    ipca_components=ipca_components,
    ipca_mean=ipca_mean,
    scaler_mean=scaler.mean_.astype(np.float32, copy=False),
    scaler_scale=scaler.scale_.astype(np.float32, copy=False),
    explained_variance=ipca.explained_variance_.astype(np.float32, copy=False),
    explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32, copy=False),
    segment_sec=float(args.segment_sec),
    clip_sec=float(args.clip_sec),
    mels_start=int(args.mels_start),
    mels_end=int(args.mels_end),
    downsample_target=int(args.downsample_target),
    n_mels=int(args.n_mels),
    )
    
if __name__ == "__main__":
    main()