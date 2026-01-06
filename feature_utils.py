import numpy as np 
import torch
import torch.nn.functional as F

# Loads .npz feature files and return feature matrix S = (T, B).
def load_features_TxB(files, n_mels=128, key="feature"):
    rows = []
    for fp in files:
        with np.load(fp, mmap_mode="r") as s:
            if key not in s.files:
                return None

            S = np.squeeze(s[key]) # Drops channel axis: (1, 128, T) -> (128, T).

            if S.ndim == 1:
                raise ValueError(f"{fp} has 1D feature {S.shape}")

            # Put time on axis 0, mel on axis 1
            if S.shape[0] == n_mels:
                S = S.T
            elif S.shape[1] == n_mels:
                pass
            else:
                raise ValueError(f"{fp}: unexpected shape {S.shape}")

            rows.append(S)

    S = np.vstack(rows)
    ids = [fp.name for fp in files]

    return S, ids

def load_feature_BxT(filepath, n_mels, key="feature"):
    with np.load(filepath, mmap_mode="r") as s:
        if key not in s.files:
            return None

        S = np.squeeze(s[key])  # Drops channel axis: (1, 128, T) -> (128, T).

        if S.ndim == 1:
            raise ValueError(f"{filepath} has 1D feature {S.shape}")

        # Shape to (B, T).
        if S.shape[0] == n_mels:
            return S 
        elif S.shape[1] == n_mels:
            return S.T
        else:
            raise ValueError(f"{filepath}: unexpected shape {S.shape}")
        
def find_min_T(files, n_mels, key="feature"):
    Ts = []
    valid_files = []
    for fp in files:
        S = load_feature_BxT(fp, n_mels, key)
        if S is None:
            continue 
        Ts.append(S.shape[1])
        valid_files.append(fp)
    if not Ts:
        raise FileNotFoundError("No valid feature files found.")
    min_T = int(min(Ts))

    return min_T, valid_files

def crop_to_T(S, T_target, mode="left"):
    B, T = S.shape
    if T == T_target:
        return S
    elif T < T_target:
        raise ValueError(f"Cannot crop to larger T_target {T_target} from T {T}")

    # Crop the sequence.
    if mode == "left":
        start = 0
    elif mode == "right":
        start = T - T_target
    elif mode == "center":
        start = (T - T_target) // 2
    else:
        raise ValueError(f"Unknown crop mode {mode}")

    # Crop. 
    return S[:, start:start + T_target]

def build_feature_matrix_TxB(files, n_mels, key="feature", crop_mode="left"):
    min_T, valid_files = find_min_T(files, n_mels, key)

    X_time_major = []
    # X_mel_major = []
    ids = []

    for fp in valid_files:
        S = load_feature_BxT(fp, n_mels, key)
        if S is None:
            continue 
        S_cropped = crop_to_T(S, min_T, crop_mode)
        X_time_major.append(S_cropped.T)  # (T, B)
        # X_mel_major.append(S_cropped)      # (B, T)
        ids.append(fp.name)

    X_time_major = np.stack(X_time_major, axis=0) # (N, T, B)
    # X_mel_major = np.stack(X_mel_major, axis=0) # (N, B, T)

    result = {
        "min_T": min_T,
        "X_time_major": X_time_major,
        "ids": ids,
        # "X_mel_major": X_mel_major,
    }

    return result 

def downsample_time_avgpool_from_db(S_db, T_target, *, ref=1.0, floor_db=None):
    """Downsample log-mel spectrogram S_db = (M, T) to T_target time frames using average pooling.

    Since E(log) != log(E), we need to convert back to linear spectrogram before downsampling.

    Args:
        S_db: Log-mel spectrogram (M, T).
        T_target: Target number of time frames.
        ref: Reference value for dB conversion.
        floor_db: Floor value in dB to avoid -inf.

    Returns:
        Downsampled log-mel spectrogram (M, T_target).
    """
    x = torch.as_tensor(S_db, dtype=torch.float32)

    unbatched = (x.dim() == 2)
    if unbatched:
        x = x.unsqueeze(0)  # Shape: (1, M, T)

    if floor_db is not None:
        x = torch.maximum(x, torch.tensor(floor_db, dtype=x.dtype, device=x.device))

    # dB to linear power.
    P = ref * torch.pow(10.0, x / 10.0) # (B, M, T).

    # Average pool downsampling over time.
    P_ds = F.adaptive_avg_pool1d(P, T_target) # (B, M, T_target).
    
    # Safeguard against log(0).
    eps = 1e-10
    P_ds = torch.clamp(P_ds, min=eps)

    # Linear power to dB.
    X_db = 10.0 * torch.log10(P_ds / ref)  # (B, M, T_target).

    if floor_db is not None:
        X_db = torch.maximum(X_db, torch.tensor(floor_db, dtype=X_db.dtype, device=X_db.device))

    if unbatched:
        X_db = X_db.squeeze(0)  # Shape: (M, T_target).

    return X_db