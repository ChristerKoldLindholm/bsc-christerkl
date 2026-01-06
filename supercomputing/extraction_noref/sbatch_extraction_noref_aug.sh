#!/usr/bin/env bash
#SBATCH --job-name=logmel_extrct_noref
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=14
#SBATCH --mem=24G
#SBATCH -p gpu --gres=gpu:titanrtx:2

set -eo pipefail

# CONFIGS.
CONDA_ENV="venv_stream_py311"
OUT_NAME="Aug_6229"
DATA_DIR="$HOME/original_data/${OUT_NAME}"
OUT_DIR="$HOME/second_extraction/${OUT_NAME}"

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
echo "[info] Running log-mel feature extraction..."
srun --ntasks=1 python run_extraction_noref.py \
  --input-root "$DATA_DIR" \
  --output-root "$OUT_DIR"

echo "[done] Features written to: $OUT_DIR"
