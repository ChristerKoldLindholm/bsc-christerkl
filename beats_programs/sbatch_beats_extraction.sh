#!/usr/bin/env bash
#SBATCH --job-name=beats_extract
#SBATCH --ntasks=1
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=36G
#SBATCH -p gpu --gres=gpu:titanrtx:1

set -eo pipefail

# CONFIGS.
CONDA_ENV="venv_stream_py311"
OUT_NAME="Tracy_6230"
DATA_DIR="$HOME/original_data/${OUT_NAME}"
OUT_DIR="$HOME/beats_extraction/${OUT_NAME}"
CKPT_PATH="$HOME/beats_model_weights/BEATs_iter3_plus_AS2M.pt"

echo "[node] $(hostname)"
echo "[wd]   $(pwd)"

# Load anaconda3.
module load anaconda3/2024.10-py3.12.7
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
python -c "import sys; print('[python]', sys.executable, sys.version)"

python - <<'PY'
import sys, pkgutil
print("[python]", sys.executable, sys.version)
print("[has torch?]", pkgutil.find_loader("torch") is not None)
PY

mkdir -p "${OUT_DIR}"

# (2) Run Python driver.
echo "[info] Running BEATs feature extraction..."
srun --ntasks=1 python run_beats_extraction_slurm.py \
  --input-root "$DATA_DIR" \
  --output-root "$OUT_DIR" \
  --target-sr 16000 \
  --ckpt-path "$CKPT_PATH" \
  --batch-size 2

echo "[done] Features written to: $OUT_DIR"
