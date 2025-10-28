#!/usr/bin/env bash
#SBATCH --job-name=logmel_erda
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -p gpu --gres=gpu:titanrtx:4

set -euo pipefail

# CONFIGS
CONDA_ENV="venv_stream"
ERDA_MNT="$HOME/erda_bsc_christerkl"
ERDA_SUBDIR="dryad_hydroacoustic_inglefield"
OUT_DIR="$HOME/logmel_outputs_${SLURM_JOB_ID:-manual}"
ERDA_OUT="$ERDA_MNT/logmel_outputs_${SLURM_JOB_ID:-manual}"

echo "[node] $(hostname)"
echo "[wd]   $(pwd)"

# Load anaconda3.
module load anaconda3/2024.10-py3.12.7 || true 
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "[warn] conda not found in PATH after loading module."
fi

# (1) Mount ERDA.
echo "[info] Mounting ERDA..."
"$HOME/bin/mount_erda.sh"

mkdir -p "${OUT_DIR}"

# (2) Run Python driver.
echo "[info] Running log-mel feature extraction..."
srun --ntasks=1 python3 run_logmel_slurm.py \
  --input-root "$ERDA_MNT/$ERDA_SUBDIR" \
  --output-root "$OUT_DIR"

mkdir -p "$ERDA_OUT"
echo "[info] Copying outputs to ERDA..."
rsync -avP "$OUT_DIR/" "$ERDA_OUT/"

# (3) Unmount ERDA.
echo "[info] Unmounting ERDA..."
"$HOME/bin/unmount_erda.sh"

echo "[done] Features written to: $OUT_DIR"
echo "[done] Features synced to ERDA: $ERDA_OUT"
