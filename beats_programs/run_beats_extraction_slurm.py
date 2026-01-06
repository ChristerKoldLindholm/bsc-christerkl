#!/usr/bin/env python
import argparse
import csv
import numpy as np
from pathlib import Path
import sys
import torch
import torchaudio
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
utils_path = ROOT / "bsc_project"
sys.path.insert(0, str(utils_path))
import utils
import configs 

def _to_int(x):
    if isinstance(x, int):
        return x
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)

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

def load_beats_model(device: torch.device, ckpt_path: Path):
    beats_path = ROOT / "unilm" / "beats" 
    sys.path.insert(0, str(beats_path.resolve()))
    from BEATs import BEATs, BEATsConfig

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    
    return model 

# Utility to get dataset paths.
def dataset_paths(ds):
    for attr in ("paths"):
        if hasattr(ds, attr):
            xs = getattr(ds, attr)
            return [Path(p) for p in xs]
    raise AttributeError("Dataset has no attribute 'paths'.")

# Utility to remove files which already had features extracted.
def ts_key_from_name(name: str) -> str:
    # Example: "6230.221003000000" -> "6230221003000000"
    s = Path(name).stem 
    s = s.replace(".", "")
    return int(s)

def pick_n_past_future(paths, center_name: str, n: int):
    kc = ts_key_from_name(center_name)
    keyed = []

    for p in map(Path, paths):
        try:
            k = ts_key_from_name(p.stem)
        except Exception:
            continue
        keyed.append((k, p))

    if not keyed:
        return [], []
    
    keyed.sort(key=lambda x: x[0])
    keys = np.array([k for k, _ in keyed], dtype=np.int64)

    idx = int(np.searchsorted(keys, kc, side="left"))

    past_slices = keyed[max(0, idx - n):idx]
    past = [p for _, p in past_slices]

    future_slices = keyed[idx: idx + 1 + n]
    future = [p for _, p in future_slices]
    return past, future

def existing_feature_stems(existing_features_root: Path):
    stems = set()
    if existing_features_root is None:
        return stems
    if not existing_features_root.exists():
        return stems 
    for p in existing_features_root.rglob("*.npz"):
        stems.add(p.stem)
    return stems

def main():
    # Slurm script arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, help="Data root folder.")
    ap.add_argument("--output-root", required=True, help="Output folder to write .npz features + index.csv.")
    ap.add_argument("--target-sr", type=int, default=16000, help="Target sample rate for audio.")
    ap.add_argument("--ckpt-path", type=Path, required=True, help="Path to the BEATs model checkpoint.")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size for dataloader.")

    ap.add_argument("--existing-features-root", type=Path, default=None
                    , help="Path to existing features root directory.")
    
    ap.add_argument("--range-a", type=str, default=None, help="Start filename to select data range.")
    ap.add_argument("--range-b", type=str, default=None, help="End filename to select data range.")

    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader setup.
    skip_secs = 5
    batch_size = args.batch_size
    dataset = utils.AudioDataset(input_root, target_sr=16_000, skip_secs=skip_secs, mode="crop", max_secs=None)
    # Check for already processed features to skip.
    ds_paths = dataset_paths(dataset)
    past, future = pick_n_past_future(ds_paths, center_name="6230.221003000000", n=256)
    selected_paths = past + future
    selected_set = {p.resolve() for p in selected_paths}
    keep_idx = [i for i, p in enumerate(ds_paths) if p.resolve() in selected_set]
    subset = Subset(dataset, keep_idx)
    # Load dataset.
    # start_idx = 0 # Create a data subset to skip to later recordings.
    # end_idx = len(dataset)
    # subset = Subset(dataset, range(start_idx, end_idx))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=utils.max_len_collate) # Only shuffle data when training.

    print("len(dataset) =", len(dataset))
    print("loader batches =", len(loader))

    # BEATs transformation pipeline.
    model = load_beats_model(device, ckpt_path)

    index_rows = []
    torch.set_grad_enabled(False)

    processed = 0
    for i, batch in enumerate(loader, 1):
        waveforms = batch["waveforms"]
        srs = batch["sample_rates"]
        paths = batch["paths"]

        if waveforms.ndim == 2:
            # Add channel dimension if missing: (B, T) -> (B, 1, T).
            waveforms = waveforms.unsqueeze(1)  
        
        B = waveforms.size(0)
        for b in range(B):
            wav_path = Path(paths[b])
            wf = waveforms[b]
            sr_val = _to_int(srs[b])

            try:
                wf = wf.to(device=device, dtype=torch.float32)
                feat, padding_mask = model.extract_features(wf, padding_mask=None)
                
                # Prepare output directory.
                try: 
                    rel = wav_path.relative_to(input_root)
                    out_dir = output_root / rel.parent
                except ValueError:
                    out_dir = output_root

                out_dir.mkdir(parents=True, exist_ok=True)
                # Write a feature as .npz for each file.
                out_path = out_dir / (wav_path.stem + ".npz")
                # Save compressed numpy data.
                np.savez_compressed(
                    str(out_path),
                    feature=feat.detach().cpu().numpy(),
                    sr=sr_val,
                    source_path=str(wav_path)
                )

                index_rows.append({
                    "source_path": str(wav_path),
                    "feature_path": str(out_path),
                    "sr": sr_val,
                    "shape": list(feat.shape)
                })

            except Exception as e:
                print(f"[error] {wav_path}: {e}", file=sys.stderr)
        
        processed += B 
        if i % 10 == 0:
            print(f"[info] Processed {processed} / {len(subset)} files.")

    print(f"[info] Processed features for {len(index_rows)} files.")

    # Write an index CSV file.
    index_csv = output_root / "features_index.csv"
    with index_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_path", "feature_path", "sr", "shape"])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"[done] Wrote {len(index_rows)} feature files.")
    print(f"[index] {index_csv}")
    
if __name__ == "__main__":
    main()