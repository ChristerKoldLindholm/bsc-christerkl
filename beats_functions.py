import torch
from pathlib import Path
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans


def pick_random_files(val_files, dataset_root, n_random=20, seed=0, exts=(".wav")):
    val_files = [Path(p) for p in val_files]
    all_wavs = [p for p in dataset_root.rglob("*") if p.suffix in exts]
    val_set = {p.resolve() for p in val_files}
    candidates = [p for p in all_wavs if p.resolve() not in val_set]
    rng = np.random.default_rng(seed)
    random_files = rng.choice(candidates, size=min(n_random, len(candidates)), replace=False)
    files = val_files + list(random_files)
    return files

def segment_wav_mono16k(wav, sr, seg_sec=5.0, hop_sec=2.5):
    seg_len = int(seg_sec * sr)
    hop_len = int(hop_sec * sr)

    T = wav.shape[1]
    segments, seg_times = [], []
    start = 0
    while start + seg_len <= T:
        end = start + seg_len
        seg = wav[:, start:end]
        segments.append(seg.squeeze(0))
        seg_times.append((start / sr, end / sr))
        start += hop_len

    if not segments:
        return None, None
    # (N_segs, seg_len), list([start_sec, end_sec]).
    return torch.stack(segments, dim=0), seg_times 

# Create BEATs embeddings from audio files.
def beats_embeddings_from_files(model, files, device, load_audio_fn
                               , seg_sec=5.0, hop_sec=5.0, batch_size=8
                               , mode="pool"):
    all_embeds = []
    all_meta = [] # (file, t0, t1, seg_idx).

    for fp in files:
        wav, sr = load_audio_fn(fp, target_sr=16000)
        segments, seg_times = segment_wav_mono16k(wav, sr, seg_sec, hop_sec)
        if segments is None:
            continue

        N_segs = segments.shape[0]
        for i in range(0, N_segs, batch_size):
            batch = segments[i:i+batch_size].to(device) # (B, T).
            x, _ = model.extract_features(batch, padding_mask=None)
            x = x.cpu()
            B, T_frames, C = x.shape

            if mode == "pool":
                clip = x.mean(dim=1)  # (B, C_dim). Collapses time dimension.
                all_embeds.append(clip)
                for j in range(B):
                    t0, t1 = seg_times[i + j]
                    all_meta.append((fp.name, t0, t1, i + j))

            elif mode == "flat_segment":
                clip = x.reshape(B, T_frames * C)  # (B, T_frames * C_dim).
                all_embeds.append(clip)
                for j in range(B):
                    t0, t1 = seg_times[i + j]
                    all_meta.append((fp.name, t0, t1, i + j))

            elif mode == "frames":
                frames = x.reshape(B * T_frames, C)  # (B * T_frames, C_dim).
                all_embeds.append(frames)
                for j in range(B):
                    seg_idx = i + j
                    t0, t1 = seg_times[seg_idx]
                    for f in range(T_frames):
                        all_meta.append((fp.name, t0, t1, seg_idx, f))

    if not all_embeds:
        return None, None
    return torch.cat(all_embeds, dim=0), all_meta # (N_total, C_dim), meta list.

# Create BEATs segments from .npz embedding files.
def beats_segs_from_embeddings(npz_files, seg_sec=5.0, hop_sec=5.0, mode="pool"
                               , device="cpu", use_source_path=True
                               , file_duration_sec=265.0):
    
    """Load BEATs embeddings from .npz files and return segments with metadata."""
    
    all_embeds = []
    all_meta = []  # (file, t0, t1, seg_idx).
    seg_global_idx = 0

    for npz_path in npz_files:
        with np.load(npz_path, allow_pickle=False) as z:
            feat = z["feature"]  # (T_frames, C_dim) or (C_dim,).
            src = None 
            if use_source_path and "source_path" in z:
                src = str(z["source_path"])
            file_id = src if src is not None else npz_path.name

        if feat.ndim == 3 and feat.shape[0] == 1:
            feat = feat[0] # (T_frames, C_dim).
        if feat.ndim != 2:
            raise ValueError(f"Unexpected feature shape in {npz_path}: {feat.shape}")
        
        T_frames, C = feat.shape

        frame_hop = file_duration_sec / T_frames

        seg_frames = int(seg_sec / frame_hop)
        hop_frames = int(hop_sec / frame_hop)
        if seg_frames <= 0 or hop_frames <= 0:
            raise ValueError(f"Segment or hop frames <= 0 in {npz_path}: seg_frames={seg_frames}, hop_frames={hop_frames}")
        if T_frames < seg_frames:
            continue  # Skip too short files.
        
        start_frames = range(0, T_frames - seg_frames + 1, hop_frames)
        for start_f in start_frames:
            end_f = start_f + seg_frames
            x = torch.from_numpy(feat[start_f:end_f,]).to(device) # (seg_frames, C).

            t0 = start_f * frame_hop
            t1 = end_f * frame_hop

            if mode == "pool":
                clip = x.mean(dim=0)  # (C_dim,).
                all_embeds.append(clip.unsqueeze(0))  # (1, C_dim).
                all_meta.append((file_id, t0, t1, seg_global_idx, start_f, end_f))

            elif mode == "flat_segment":
                clip = x.reshape(1, seg_frames * C)  # (1, seg_frames * C_dim).
                all_embeds.append(clip)  # (1, seg_frames * C_dim).
                all_meta.append((file_id, t0, t1, seg_global_idx, start_f, end_f))

            elif mode == "frames":
                all_embeds.append(x)  # (seg_frames, C_dim).
                for f in range(seg_frames):
                    all_meta.append((file_id, t0, t1, seg_global_idx, start_f + f))

            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            seg_global_idx += 1
            
    if not all_embeds:
        return None, None
    
    embeds = torch.cat(all_embeds, dim=0) # (N_total, C_dim) or (N_total, seg_frames * C_dim).
    return embeds, all_meta

def iter_beats_segments_npz(npz_files, seg_sec=5.0, hop_sec=5.0
                        , mode="flat_segment", device="cpu", file_duration_sec=265.0
                        , batch_size=256,):
    batch_X = []
    batch_meta = []
    seg_global_idx = 0

    for npz_path in npz_files:
        with np.load(npz_path, allow_pickle=False) as z:
            feat = z["feature"]
            file_id = str(z["source_path"]) if "source_path" in z.files else npz_path.name

        if feat.ndim == 3 and feat.shape[0] == 1:
            feat = feat[0]
        if feat.ndim != 2:
            raise ValueError(f"Unexpected feature shape in {npz_path}: {feat.shape}")
        
        T_frames, C = feat.shape

        frame_hop = file_duration_sec / T_frames
        seg_frames = int(round(seg_sec / frame_hop))
        hop_frames = int(round(hop_sec / frame_hop))
        if seg_frames <= 0 or hop_frames <= 0 or T_frames < seg_frames:
            continue

        for start_f in range(0, T_frames - seg_frames + 1, hop_frames):
            end_f = start_f + seg_frames
            x = torch.from_numpy(feat[start_f:end_f]).to(device)  # (seg_frames, C)

            t0 = start_f * frame_hop
            t1 = end_f * frame_hop

            if mode == "pool":
                clip = x.mean(dim=0)  # (C,)
                X_row = clip.unsqueeze(0)  # (1, C)
            elif mode == "flat_segment":
                X_row = x.reshape(1, seg_frames * C)  # (1, seg_frames*C)
            else:
                raise ValueError(mode)

            batch_X.append(X_row.cpu())
            batch_meta.append((file_id, t0, t1, seg_global_idx, start_f, end_f))
            seg_global_idx += 1

            if len(batch_X) >= batch_size:
                yield torch.cat(batch_X, dim=0).numpy(), batch_meta
                batch_X, batch_meta = [], []

    if batch_X:
        yield torch.cat(batch_X, dim=0).numpy(), batch_meta

def elbow_streaming_ipca_kmeans(files,
    n_comps_list=(2,4,8,16,32,64,128),
    k_values=range(1, 16),
    seg_sec=5.0,
    hop_sec=5.0,
    mode="flat_segment",
    batch_size=256,
    random_state=104,
    file_duration_sec=265.0,
    dtype=np.float32,
):
    inertias_ncomps = {}

    for n_comps in n_comps_list:
        # 1) Fit IPCA on streaming batches (no scaling here; add scaler if you want)
        ipca = IncrementalPCA(n_components=n_comps, batch_size=batch_size)

        for Xb, _ in iter_beats_segments_npz(
            files, seg_sec=seg_sec, hop_sec=hop_sec, mode=mode,
            batch_size=batch_size, file_duration_sec=file_duration_sec
        ):
            Xb = Xb.astype(dtype, copy=False)
            ipca.partial_fit(Xb)

        # 2) Fit all k models in parallel on PCA features (one pass)
        models = {
            k: MiniBatchKMeans(
                n_clusters=k, random_state=random_state, batch_size=batch_size,
                n_init="auto"
            )
            for k in k_values
        }

        for Xb, _ in iter_beats_segments_npz(
            files, seg_sec=seg_sec, hop_sec=hop_sec, mode=mode,
            batch_size=batch_size, file_duration_sec=file_duration_sec
        ):
            Xb = Xb.astype(dtype, copy=False)
            Zb = ipca.transform(Xb).astype(dtype, copy=False)
            for km in models.values():
                km.partial_fit(Zb)

        # 3) Compute true inertia (SSE) in one more pass
        inertias = {k: 0.0 for k in k_values}

        for Xb, _ in iter_beats_segments_npz(
            files, seg_sec=seg_sec, hop_sec=hop_sec, mode=mode,
            batch_size=batch_size, file_duration_sec=file_duration_sec
        ):
            Xb = Xb.astype(dtype, copy=False)
            Zb = ipca.transform(Xb).astype(dtype, copy=False)

            for k, km in models.items():
                # distances: (B, k). squared=True gives squared Euclidean distances.
                d2 = km.transform(Zb) ** 2
                inertias[k] += np.min(d2, axis=1).sum()

        inertias_ncomps[n_comps] = [inertias[k] for k in k_values]

    return inertias_ncomps