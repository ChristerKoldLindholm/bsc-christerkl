#!/usr/bin/env py
import argparse
import csv
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import utils
import configs 

def _to_int(x):
    if isinstance(x, int):
        return x
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, help="ERDA root folder with audio files.")
    ap.add_argument("--output-root", required=True, help="Output folder to write .npz features + index.csv.")
    ap.add_argument("--subset-len", type=int, default=0, help="Choose a subset of data.")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    subset_len = max(0, int(args.subset_len))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader setup.
    skip_secs = 5
    batch_size = 2
    dataset = utils.AudioDataset(input_root, target_sr=64_000, skip_secs=skip_secs, mode="crop", max_secs=None)
    start_idx = 0 # Create a data subset to skip to later recordings.
    if subset_len:
        end_idx = subset_len
    else:
        end_idx = len(dataset)
    subset = Subset(dataset, range(start_idx, end_idx))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=utils.max_len_collate) # Only shuffle data when training.

    max_points = None # Used by tensors_to_array().

    # Transformation pipeline.
    specgram_config = configs.get_specgram_config()
    logmel_transf = utils.PipelineSpecgram(specgram_config=specgram_config).to(device)
    if hasattr(logmel_transf, "eval"): # Set to evaluation mode. 
        logmel_transf.eval()

    index_rows = []
    torch.set_grad_enabled(False)

    processed = 0
    for i, batch in enumerate(loader, 1):
        waveforms = batch["waveforms"]
        srs = batch["sample_rates"]
        paths = batch["paths"]
        # lengths = batch.get("lengths", None)

        if waveforms.ndim == 2:
            # Add channel dimension if missing: (B, T) -> (B, 1, T).
            waveforms = waveforms.unsqueeze(1)  
        
        B = waveforms.size(0)
        for b in range(B):
            wav_path = Path(paths[b])
            wf = waveforms[b]
            sr_val = _to_int(srs[b])

            try:
                wf = wf.to(dtype=torch.float32, device=device)
                feat = logmel_transf(wf)

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
            print(f"[info] Processed {i * batch_size} / {len(subset)} files.")

    print(f"[info] Processed features for {len(index_rows)} files.")

    # Write an index CSV file.
    index_csv = output_root / "features_index.csv"
    with index_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_path", "feature_path", "sr", "shape"])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"[done] Wrote {len(index_rows)} feature files.")
    print(f"[index] {index_csv}")
    
    print("[info] Starting tensors_to_array()")
    X, ids = utils.tensors_to_array(dataloader=loader, transform=logmel_transf, max_pts=max_points, device=device)

    X_folder = output_root / "numpy_arrays" 
    X_folder.mkdir(parents=True, exist_ok=True)
    X_name = "logmel_features.npz"

    np.savez_compressed(X_folder / X_name, X=X, ids=ids)

    meta = {
        "specgram_config": str(specgram_config),
        "max_points": max_points,
        "dtype": str(X.dtype),
        "shape": list(X.shape),
        "created": datetime.now().isoformat()
    }
    with open(str(X_folder / X_name) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()