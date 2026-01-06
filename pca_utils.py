import numpy as np
from pathlib import Path
import feature_utils as futils
from sklearn.preprocessing import StandardScaler


def list_feature_files(folder: Path):
    files = sorted(folder.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {folder}")
    return files 

def stream_segments(files, batch_segments:int, segment_sec:float=5.0, clip_sec:float=270.0,
                    mels_start:int=9, mels_end:int=60, key:str="feature", downsample_T:int=200):
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
            s = s[mels_start:mels_end, :]  # Drop lowest mel bins.
            if s.ndim != 2:
                continue
            M, T = s.shape

            frames_per_segment = int(round(T * (segment_sec / clip_sec)))
            if frames_per_segment <= 0 or frames_per_segment > T:
                continue

            seg_idx = 0
            for start in range(0, T - frames_per_segment + 1, frames_per_segment):
                end = start + frames_per_segment
                seg = s[:, start:end]
                seg = futils.downsample_time_avgpool_from_db(seg, T_target=downsample_T, ref=1.0).numpy()
                v = seg.astype(np.float32, copy=False).ravel()[None, :]  # (1, D)

                batch.append(v)
                meta_batch.append(dict(
                    file_path=str(fp),
                    segment_idx=seg_idx,
                    start_frame=int(start),
                    end_frame=int(end),
                ))
                seg_idx += 1

                if len(batch) == batch_segments:
                    Xb = np.vstack(batch).astype(np.float32, copy=False)
                    yield Xb, meta_batch
                    batch, meta_batch = [], []

    if batch:
        Xb = np.vstack(batch).astype(np.float32, copy=False)
        yield Xb, meta_batch

# Pseudo-global scaler computation.
def compute_pseudo_global_scaler(
    data_root: Path,
    target_segments: int = 1000,
    batch_segments: int = 128,
    segment_sec: float = 5.0,
    clip_sec: float = 270.0,
    mels_start: int = 9,
    mels_end: int = 60,
    key: str = "feature",
    seed: int = 104,
    downsample_T: int = 300,
):
    files = list_feature_files(data_root)
    rng = np.random.default_rng(seed)
    rng.shuffle(files)  # Randomly selected files.

    scaler = StandardScaler(with_mean=True, with_std=True)

    total_seen = 0
    for Xb, meta_batch in stream_segments(
        files,
        batch_segments=batch_segments,
        segment_sec=segment_sec,
        clip_sec=clip_sec,
        mels_start=mels_start,
        mels_end=mels_end,
        key=key,
        downsample_T=downsample_T,
    ):
        # Restrict to exactly target_segments if last batch overshoots.
        remaining = target_segments - total_seen
        if Xb.shape[0] > remaining:
            Xb = Xb[:remaining, :]

        scaler.partial_fit(Xb)
        total_seen += Xb.shape[0]
        if total_seen >= target_segments:
            break

    scaler_mean = scaler.mean_.astype(np.float32)
    scaler_scale = scaler.scale_.astype(np.float32)
    scaler_scale[scaler_scale == 0] = 1.0  # safety.

    print(f"Pseudo-global scaler trained on {total_seen} segments")
    print("D =", scaler_mean.shape[0])

    return scaler_mean, scaler_scale

def build_segment_matrix(
    files,
    scaler_mean,
    scaler_scale,
    max_segments: int,
    batch_segments: int = 128,
    segment_sec: float = 5.0,
    clip_sec: float = 265.0,
    mels_start: int = 9,
    mels_end: int = 60,
    key: str = "feature",
    downsample_T: int = 200,
):
    """Return (X, Z) where:
       X: (N, D) original flattened segments
       Z: (N, D) standardized segments
    """
    X_list = []
    Z_list = []
    total = 0

    for Xb, meta_batch in stream_segments(
        files,
        batch_segments=batch_segments,
        segment_sec=segment_sec,
        clip_sec=clip_sec,
        mels_start=mels_start,
        mels_end=mels_end,
        key=key,
        downsample_T=downsample_T,
    ):
        X_list.append(Xb)
        Zb = (Xb - scaler_mean) / scaler_scale
        Z_list.append(Zb)

        total += Xb.shape[0]
        if total >= max_segments:
            break

    X = np.vstack(X_list)[:max_segments].astype(np.float32, copy=False)
    Z = np.vstack(Z_list)[:max_segments].astype(np.float32, copy=False)
    return X, Z

def reconstruction_error_for_K(X: np.ndarray, Z: np.ndarray, W: np.ndarray
                               , scaler_mean: np.ndarray, scaler_scale: np.ndarray, K: int,
):
    """
    X: (N, D) original segments.
    Z: (N, D) standardized segments.
    W: (K_max, D) PCA components.
    """
    W_K = W[:K, :] # (K, D)
    Y = Z @ W_K.T # (N, K)
    Z_hat = Y @ W_K # (N, D)
    X_hat = Z_hat * scaler_scale + scaler_mean
    err = np.mean((X - X_hat)**2, axis=1)
    return err, X_hat

def reconstruct_narwhal_spectrogram(narwhal_file: Path, scaler_mean: np.ndarray, scaler_scale: np.ndarray
                                    , W_K: np.ndarray, segment_sec: float = 5.0, clip_sec: float = 265.0
                                    ,mels_start: int = 9, mels_end: int = 60, key: str = "feature",
):
    with np.load(narwhal_file, mmap_mode="r") as z:
        s = np.squeeze(z[key])
        s = s[mels_start:mels_end, :]
        if s.ndim != 2:
            raise ValueError("Expected 2D spectrogram")
        M, T = s.shape

        frames_per_segment = int(round(T * (segment_sec / clip_sec)))
        if frames_per_segment <= 0 or frames_per_segment > T:
            raise ValueError("Bad frames_per_segment")

        S_hat = np.zeros_like(s, dtype=np.float32)

        D = scaler_mean.shape[0]
        if M * frames_per_segment != D:
            raise ValueError(f"Dimension mismatch: M*frames_per_segment={M*frames_per_segment} != D={D}")

        for seg_idx, start in enumerate(range(0, T - frames_per_segment + 1, frames_per_segment)):
            end = start + frames_per_segment

            seg = s[:, start:end] # (M, frames_per_segment)
            x = seg.ravel().astype(np.float32) # (D,)

            z_vec = (x - scaler_mean) / scaler_scale

            y = W_K @ z_vec # (K,)
            z_hat = W_K.T @ y # (D,)

            x_hat = z_hat * scaler_scale + scaler_mean # (D,)

            seg_hat = x_hat.reshape(M, frames_per_segment)
            S_hat[:, start:end] = seg_hat

    return s, S_hat

def reconstruct_segment(seg, n_components, scaler_mean
                        , scaler_scale, pca_mean, pca_components):
    """
    seg: (M, T_seg).
    n_components: k <= pca_components.shape[0]
    """
    M, T_seg = seg.shape
    x = seg.astype(np.float32).ravel()[None, :] # (1, D)
    # (1) Standardize using population scaler.
    x_sc = (x - scaler_mean) / scaler_scale
    # (2) Project to first k PCs.
    C_k = pca_components[:n_components, :] # (k, D)
    Z = (x_sc - pca_mean) @ C_k.T # (1, k)
    # (3) Reconstruct in standardized space.
    x_sc_hat = Z @ C_k + pca_mean # (1, D)
    # (4) Undo scaling.
    x_hat = x_sc_hat * scaler_scale + scaler_mean  # (1, D)
    # (5) Reshape back to spectrogram.
    return x_hat.reshape(M, T_seg)

def reconstruct_single_segment(narwhal_file: Path, scaler_mean: np.ndarray, scaler_scale: np.ndarray
                               , W_K: np.ndarray, seg_idx: int = 0, segment_sec: float = 5.0, clip_sec: float = 265.0
                               ,mels_start: int = 9, mels_end: int = 60, key: str = "feature", downsample_T: int = 300,
):
    with np.load(narwhal_file, mmap_mode="r") as z:
        s = np.squeeze(z[key])
        s = s[mels_start:mels_end, :]
        s = futils.downsample_time_avgpool_from_db(s, T_target=downsample_T, ref=1.0).numpy()
        print(s.shape)

        if s.ndim != 2:
            raise ValueError("Expected 2D spectrogram")
        M, T = s.shape

        frames_per_segment = int(round(T * (segment_sec / clip_sec)))
        if frames_per_segment <= 0 or frames_per_segment > T:
            raise ValueError("Bad frames_per_segment")
        
        D = scaler_mean.shape[0]
        if M * frames_per_segment != D:
            raise ValueError(f"Dimension mismatch: M*frames_per_segment={M*frames_per_segment} != D={D}")
        
        start = seg_idx * frames_per_segment
        end = start + frames_per_segment
        if end > T:
            raise IndexError("seg_idx out of range for this file")
        
        seg_orig = s[:, start:end]
        x = seg_orig.ravel().astype(np.float32)
        z_vec = (x - scaler_mean) / scaler_scale

        # PCA projection + reconstruction.
        y = W_K @ z_vec
        z_hat = W_K.T @ y
        x_hat = z_hat * scaler_scale + scaler_mean
        seg_hat = x_hat.reshape(M, frames_per_segment)

        t_start = seg_idx * segment_sec
        t_end = t_start + segment_sec

    return seg_orig, seg_hat, t_start, t_end

def reconstruct_from_segment_matrix(seg: np.ndarray, scaler_mean: np.ndarray, scaler_scale: np.ndarray
                                    , W_K: np.ndarray, T_target: int = 300, ref: float = 1.0,
):
    # (1) Downsample.
    seg_ds = futils.downsample_time_avgpool_from_db(seg, T_target=T_target, ref=ref).numpy() # (M, T_target).

    if seg_ds.ndim != 2:
        raise ValueError("Expected 2D segment")

    M, T_ds = seg_ds.shape
    D = scaler_mean.shape[0]
    if M * T_ds != D:
        raise ValueError(
            f"Dimension mismatch: M*T_ds={M*T_ds} != D={D}"
        )

    # (2) Flatten and standardize.
    x = seg_ds.ravel().astype(np.float32) # (D,)
    z_vec = (x - scaler_mean) / scaler_scale
    # (3) PCA projection + reconstruction
    y = W_K @ z_vec # (K,)
    z_hat = W_K.T @ y # (D,)
    x_hat = z_hat * scaler_scale + scaler_mean # (D,)

    seg_hat_ds = x_hat.reshape(M, T_ds) # (M, T_target)

    return seg_ds, seg_hat_ds