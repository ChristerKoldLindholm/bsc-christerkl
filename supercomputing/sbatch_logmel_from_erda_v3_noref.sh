#!/usr/bin/env bash
#SBATCH --job-name=logmel_erda_v2
#SBATCH --ntasks=1
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH -p gpu --gres=gpu:titanrtx:1

set -eo pipefail

# CONFIGS
CONDA_ENV="venv_stream_py311"
ERDA_MNT="$HOME/erda_bsc_christerkl"
MNT="$ERDA_MNT"
ERDA_SUBDIR="dryad_hydroacoustic_inglefield"
OUT_DIR="$HOME/logmel_outputs_${SLURM_JOB_ID:-manual}"
ERDA_OUT="$ERDA_MNT/logmel_outputs_${SLURM_JOB_ID:-manual}"
REMOTE_DIR="/erda_bsc_christerkl/dryad_hydroacoustic_inglefield"

echo "[node] $(hostname)"
echo "[wd]   $(pwd)"

# Load anaconda3.
module load anaconda3/2024.10-py3.12.7
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv_stream_py311
export PYTHONNOUSERSITE=1
python -c "import sys; print('[python]', sys.executable, sys.version)"
python - <<'PY'
import sys, pkgutil
print("[python]", sys.executable, sys.version)
print("[has torch?]", pkgutil.find_loader("torch") is not None)
PY

echo "[mount] Preparing ERDA mount at $MNT"

# # Try to unmount *unconditionally*; ignore errors so set -e doesn't kill us.
# fusermount -u "$MNT" 2>/dev/null || fusermount -uz "$MNT" 2>/dev/null || true

# # Optional: small debug
# mount | grep erda_bsc_christerkl || echo "[mount] no existing erda_bsc_christerkl mount"

mkdir -p "${OUT_DIR}"

# bash -x ./unmount_erda.sh || echo "[mount] Initial unmount_erda.sh failed (may be unmounted already)"

# (1) Mount ERDA.
echo "[info] Mounting ERDA..."
bash -x ./mount_erda.sh

# (2) Run Python driver.
echo "[info] Running log-mel feature extraction..."
srun --ntasks=1 python run_logmel_slurm_noref.py \
  --input-root "$ERDA_MNT" \
  --output-root "$OUT_DIR"

mkdir -p "$ERDA_OUT"
echo "[info] Copying outputs to ERDA..."
rsync -avP "$OUT_DIR/" "$ERDA_OUT/"

# (3) Unmount ERDA.
echo "[info] Unmounting ERDA..."
bash -x ./unmount_erda.sh

echo "[done] Features written to: $OUT_DIR"
echo "[done] Features synced to ERDA: $ERDA_OUT"
