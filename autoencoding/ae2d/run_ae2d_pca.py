import argparse
import csv
import numpy as np
import os
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler, random_split
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import silhouette_score
import heapq
from numpy.lib.format import open_memmap

import_path = Path.cwd().parents[1]
os.sys.path.insert(0, str(import_path))
import feature_utils as futils
import utils
import autoencoder_functions as aefuncs


def slurm_cpus() -> int:
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v is not None:
        try:
            return max(1, int(v))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)

def stream_npy_rows(npy_path, batch_size: int):
    """
    Streams rows from a saved .npy array.
    """
    X = np.load(npy_path, mmap_mode="r")
    N = X.shape[0]
    for i in range(0, N, batch_size):
        yield np.asarray(X[i:i+batch_size], dtype=np.float32)

@torch.no_grad()
def fit_embedding_scaler_streaming(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    pool: nn.Module,
    max_samples: int | None = None,
) -> StandardScaler:
    """
    Fits a StandardScaler on embeddings z .
    """
    model.eval()
    emb_scaler = StandardScaler(with_mean=True, with_std=True)

    seen = 0
    for Z, _meta in stream_embeddings(model, loader, device, pool):  # Z: (B, D_embed) float32
        emb_scaler.partial_fit(Z)
        seen += Z.shape[0]
        if max_samples is not None and seen >= max_samples:
            break

    return emb_scaler

@torch.no_grad()
def stream_embeddings_standardized(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    pool: nn.Module,
    emb_scaler: StandardScaler,
    max_samples: int | None = None,
):
    """
    Yields standardized embeddings Zs = (Z - mean) / scale, along with meta.
    """
    model.eval()
    seen = 0
    for Z, meta in stream_embeddings(model, loader, device, pool):
        Zs = emb_scaler.transform(Z).astype(np.float32, copy=False)
        yield Zs, meta
        seen += Z.shape[0]
        if max_samples is not None and seen >= max_samples:
            break

@torch.no_grad()
def infer_embed_dim(model: nn.Module, loader: DataLoader, device: str, pool: nn.Module) -> int:
    model.eval()
    x, _meta = next(iter(loader))
    x = x.to(device, non_blocking=True)
    _recon, z = model(x)
    z = pool(z).flatten(1)
    return int(z.shape[1])

@torch.no_grad()
def stream_embeddings(model: nn.Module, loader: DataLoader, device: str, pool: nn.Module):
    model.eval()
    for x, meta in loader:
        x = x.to(device, non_blocking=True)
        _recon, z = model(x)
        z = pool(z).flatten(1)  # (B, D_embed)
        Z = z.detach().cpu().numpy().astype(np.float32, copy=False)
        yield Z, meta

def fit_ipca_streaming(model: nn.Module, loader: DataLoader, device: str
                       , pool: nn.Module, scaler: StandardScaler, n_components: int
                       , max_samples: int | None, random_state: int, embed_scaler: StandardScaler
                       ) -> IncrementalPCA:
    ipca = IncrementalPCA(n_components=n_components)
    seen = 0
    for Z, _meta in stream_embeddings(model, loader, device, pool):
        Zs = embed_scaler.transform(Z)
        ipca.partial_fit(Zs)
        seen += Z.shape[0]
        if max_samples is not None and seen >= max_samples:
            break
    return ipca

def transform_to_pcs(Z: np.ndarray, embed_scaler: StandardScaler, ipca: IncrementalPCA, n_pc: int) -> np.ndarray:
    Zs = embed_scaler.transform(Z)
    Y = ipca.transform(Zs)  # (B, n_components).
    return Y[:, :n_pc].astype(np.float32, copy=False)

def fit_kmeans_streaming_from_embeddings_npy(
    embeddings_npy,
    k: int,
    kmeans_batch_size: int,
    reservoir_cap: int = 32000,
    random_state: int = 104,
):
    rng = np.random.default_rng(random_state)

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init="auto",
        batch_size=kmeans_batch_size,
        max_iter=100,
        random_state=random_state,
    )

    keep = []
    keep_n = 0

    for Y in stream_npy_rows(embeddings_npy, batch_size=kmeans_batch_size):
        # Y: (B,2) already your saved "pcs2"
        kmeans.partial_fit(Y)

        keep.append(Y)
        keep_n += Y.shape[0]

        if keep_n > reservoir_cap * 2:
            Ytmp = np.vstack(keep)
            idx = rng.choice(Ytmp.shape[0], reservoir_cap, replace=False)
            keep = [Ytmp[idx]]
            keep_n = keep[0].shape[0]

    sil = np.nan
    if keep_n > k:
        Ys = np.vstack(keep)
        labs = kmeans.predict(Ys)
        if np.unique(labs).size > 1:
            sil = float(silhouette_score(Ys, labs))

    return kmeans, sil

def assign_labels_and_distances_from_embeddings_npy(
    embeddings_npy: Path,
    meta_csv: Path,
    kmeans,  # Fitted MiniBatchKMeans.
    out_labels_npy: Path,
    out_dist2_npy: Path,
    out_nearest_csv: Path,
    topn: int = 25,
    batch_size: int = 8192,
):
    X = np.load(embeddings_npy, mmap_mode="r")  # (N, d)
    N, d = X.shape

    centers = kmeans.cluster_centers_.astype(np.float32, copy=False)  # (k, d)
    k = centers.shape[0]
    c2 = np.sum(centers**2, axis=1, keepdims=True).T  # (1, k)

    labels_mm = open_memmap(out_labels_npy, mode="w+", dtype=np.int32, shape=(N,))
    dist2_mm  = open_memmap(out_dist2_npy,  mode="w+", dtype=np.float32, shape=(N,))

    nearest_heaps = [[] for _ in range(k)]  # store (-dist2, row_tuple)

    with open(meta_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        i0 = 0
        while i0 < N:
            i1 = min(N, i0 + batch_size)
            Y = np.asarray(X[i0:i1], dtype=np.float32)  # (B,d)

            # squared distances: ||y-c||^2 = ||y||^2 + ||c||^2 - 2 y·c
            x2 = np.sum(Y**2, axis=1, keepdims=True)        # (B,1)
            xc = Y @ centers.T                               # (B,k)
            d2_all = x2 + c2 - 2.0 * xc                      # (B,k)

            labs = np.argmin(d2_all, axis=1).astype(np.int32)
            d2   = d2_all[np.arange(labs.size), labs].astype(np.float32, copy=False)

            labels_mm[i0:i1] = labs
            dist2_mm[i0:i1]  = d2

            # consume exactly B rows from CSV
            B = i1 - i0
            for bi in range(B):
                row = next(reader)  # raises if CSV length mismatches N
                cl = int(labs[bi])
                d2v = float(d2[bi])

                # keep lightweight tuple to avoid huge RAM
                item = (-d2v, i0 + bi, row["file"], int(row["start"]), int(row["end"]),
                        float(Y[bi, 0]) if d >= 1 else np.nan,
                        float(Y[bi, 1]) if d >= 2 else np.nan)

                h = nearest_heaps[cl]
                if len(h) < topn:
                    heapq.heappush(h, item)
                else:
                    if -h[0][0] > d2v:
                        heapq.heapreplace(h, item)

            i0 = i1

    del labels_mm
    del dist2_mm

    with open(out_nearest_csv, "w", encoding="utf-8") as f:
        f.write("cluster,rank,dist2,global_idx,file,start,end,pc1,pc2\n")
        for cl in range(k):
            items = sorted([(-neg, *rest) for (neg, *rest) in nearest_heaps[cl]], key=lambda x: x[0])
            for r, (d2v, global_idx, file, start, end, pc1, pc2) in enumerate(items, start=1):
                f.write(f"{cl},{r},{d2v:.6f},{global_idx},{file},{start},{end},{pc1},{pc2}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory with input data")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to save PCA results")
    ap.add_argument("--scaler_dir", type=str, required=True, help="Directory with saved scaler")
    ap.add_argument("--model_dir", type=str, required=True, help="Directory with trained AE model")
    
    ap.add_argument("--batch_size", type=int, default=256, help="Batch size for DataLoader")
    ap.add_argument("--segment_sec", type=float, default=5.0, help="Segment length in seconds")
    ap.add_argument("--T_target", type=int, default=300, help="Target number of time samples per segment")
    ap.add_argument("--n_pca_components", type=int, default=10, help="Number of PCA components to compute")
    ap.add_argument("--subset_size", type=int, default=50000, help="Subset size for PCA")
    ap.add_argument("--elbow_k_min", type=int, default=2, help="Minimum k for elbow method")
    ap.add_argument("--elbow_k_max", type=int, default=16, help="Maximum k for elbow method")
    ap.add_argument("--kmeans_k", type=int, default=8, help="Number of clusters for k-means")
    ap.add_argument("--kmeans_batch_size", type=int, default=4096, help="Batch size for k-means")
    args = ap.parse_args()

    rng = np.random.default_rng(seed=104)

    # Paths.
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir) if args.model_dir is not None else None

    # Configurations.
    device = "cpu"
    key = "feature"

    mels_start, mels_end, n_mels = 9, 61, 128
    mels_used = mels_end - mels_start
    sr = 64000
    hop_length = 512
    fps = sr / hop_length
    t_start, t_end = 25.0, 30.0
    i_start = int(t_start * fps)
    i_end = int(t_end * fps)
    segment_sec, clip_sec = args.segment_sec, 265
    T_target = args.T_target

    # Prepare dataset.
    print("Preparing dataset...")
    batch_size = args.batch_size
    all_paths = sorted(data_dir.rglob("*.npz"))
    all_files = list(all_paths)

    with np.load(all_files[0], mmap_mode="r") as z:
        arr = np.squeeze(z[key])
        if arr.ndim == 3:
            arr = arr[0]
        t_total = arr.shape[-1]
    frames_per_sec = t_total / clip_sec
    segment_frames = int(round(frames_per_sec * segment_sec)) # = (t / 265) * segment_sec.

    print(f"Total files = {len(all_files)}")
    dataset = aefuncs.SegmentLogMel2DDataset(all_files, key=key, mels_start=mels_start, mels_end=mels_end
                                             , segment_frames=segment_frames, T_target=T_target, add_channel_dim=True)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    N = len(dataset)

    print("Loading scaler from", args.scaler_dir)
    scaler = StandardScaler(with_mean=True, with_std=True)
    with np.load(Path(args.scaler_dir) / "scaler.npz") as z:
        scaler.mean_ = z["scaler_mean"]
        scaler.var_ = z["scaler_var"]
        scaler.scale_ = z["scaler_scale"]
        scaler.n_samples_seen_ = z["n_samples_seen"]
    print("Scaler shapes =", scaler.mean_.shape, scaler.scale_.shape)

    data_point, meta = next(iter(data_loader))
    print("data_point.shape =", data_point.shape) # (B, C, M = n_mels, t = segment_frames).
    print("meta: file =", meta[0][0], ", start =", meta[1][0], ", end =", meta[2][0])
    print("Data sample:", data_point[0, :, 0:5, 0:5])

    input_channels = data_point.shape[1]
    lat_dim = 2
    hid_dim = 16
    lr = 1e-3
    # 2D autoencoder: CNN.
    model = aefuncs.AE2D(input_channels=input_channels, latent_dim=lat_dim, hidden_dim=hid_dim).to(device)
    # U-Net autoencoder: input channels equals output channels when reconstructing with MSE loss.
    # model = aefuncs.AEUnet2D(in_channels=input_channels, out_channels=input_channels, base=32, bilinear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    # Load saved model.
    model.load_state_dict(torch.load(model_dir / "best_ae2d_model.pth", map_location=device))
    optimizer.load_state_dict(torch.load(model_dir / "latest_ae2d_optimizer.pth", map_location=device))
    print("Loaded model from", model_dir / "best_ae2d_model.pth")
    print("Loaded optimizer from", model_dir / "latest_ae2d_optimizer.pth")

    model.eval()
    pool = nn.AdaptiveAvgPool2d((1, 1))

    max_fit = int(args.subset_size) if args.subset_size is not None else None
    print(f"[info] max samples for fitting scaler: {max_fit}")
    standardizer = aefuncs.TorchStandardizer(scaler, device)

    embed_scaler = fit_embedding_scaler_streaming(
        model=model,
        loader=data_loader,
        device=device,
        pool=pool,
        max_samples=max_fit,
    )
    # # Fit IPCA on embeddings Z.
    # ipca = IncrementalPCA(n_components=args.n_pca_components)
    # seen = 0
    
    np.savez_compressed(
        output_dir / "embeddings_ipca.npz",
    #     ipca_components=ipca.components_.astype(np.float32, copy=False),
    #     ipca_mean=ipca.mean_.astype(np.float32, copy=False),
        scaler_mean=embed_scaler.mean_.astype(np.float32, copy=False),
        scaler_scale=embed_scaler.scale_.astype(np.float32, copy=False),
    #     explained_variance=ipca.explained_variance_.astype(np.float32, copy=False),
    #     explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32, copy=False),
    #     n_components=np.array([args.n_pca_components], dtype=np.int32),
    )
    # print("[info] IPCA fitted:", ipca.components_.shape)

    # ---------------------------------------------------------------------
    # 4) 2D PCA embeddings + meta CSV.
    data_loader_eval = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    emb_path = output_dir / "embeddings.npy"
    csv_path = output_dir / "embeddings_meta.csv"

    pcs_mm = open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(N, 2))
    write_i = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["file", "start", "end", "pc1", "pc2"])

        with torch.no_grad():
            for x, meta in data_loader_eval:
                x = x.to(device, non_blocking=True)
                x = standardizer.transform(x)
                _, z = model(x)
                z = pool(z).flatten(1)
                Z = z.detach().cpu().numpy().astype(np.float32, copy=False)

                # Y = ipca.transform(Zs)  # (B, n_components).
                pcs2 = embed_scaler.transform(Z)[:, :2].astype(np.float32, copy=False)

                B = pcs2.shape[0]
                pcs_mm[write_i:write_i + B, :] = pcs2

                files, start, end = meta
                for i in range(B):
                    writer.writerow([
                        str(files[i]), int(start[i]), int(end[i])
                        , float(pcs2[i, 0]), float(pcs2[i, 1])
                    ])
                write_i += B
    del pcs_mm
    print("[info] wrote embeddings PCs + meta")
    
    # Elbow method.
    elbow_rows = []
    for k in range(args.elbow_k_min, args.elbow_k_max + 1):
        km, sil = fit_kmeans_streaming_from_embeddings_npy(
            embeddings_npy=emb_path,
            k=k,
            kmeans_batch_size=args.kmeans_batch_size,
            reservoir_cap=32000,
            random_state=104,
        )
        elbow_rows.append((k, sil))
        print(f"[elbow] k={k} silhouette={sil:.6f}")

    elbow_out = output_dir / "elbow_silhouette.npy"
    np.save(elbow_out, np.array(elbow_rows))
    print("[info] wrote", elbow_out)

    # Final k-means.
    km_final, sil_final = fit_kmeans_streaming_from_embeddings_npy(
        embeddings_npy=emb_path,
        k=args.kmeans_k,
        kmeans_batch_size=args.kmeans_batch_size,
        reservoir_cap=20_000,
        random_state=104,
    )

    np.savez_compressed(
        output_dir / "kmeans_final.npz",
        centers=km_final.cluster_centers_.astype(np.float32, copy=False),
        k=np.array([args.kmeans_k], dtype=np.int32),
        n_pc=np.array([min(2, args.n_pca_components)], dtype=np.int32),
        silhouette=np.array([sil_final], dtype=np.float32),
    )
    print(f"[info] saved final kmeans, silhouette={sil_final:.6f}")

    assign_labels_and_distances_from_embeddings_npy(
        embeddings_npy=emb_path,
        meta_csv=csv_path,
        kmeans=km_final,
        out_labels_npy=output_dir / "segments_labels.npy",
        out_dist2_npy=output_dir / "segments_dist2.npy",
        out_nearest_csv=output_dir / "nearest_per_cluster.csv",
        topn=25,
        batch_size=args.kmeans_batch_size,
    )

    print("[info] wrote PCs/labels/dist2/meta/nearest")


if __name__ == "__main__":
    main()