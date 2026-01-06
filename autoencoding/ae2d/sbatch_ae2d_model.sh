#!/usr/bin/env bash
#SBATCH --job-name=ae2d_model
#SBATCH --ntasks=1
#SBATCH --time=38:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH -p gpu --gres=gpu:titanrtx:1

set -eo pipefail

# CONFIGS
CONDA_ENV="venv_stream_py311"
IN_DIR="$HOME/extraction_uwref/Tracy_6230"
SEC_IN_DIR="$HOME/extraction_uwref/Aug_6229"
TER_IN_DIR="$HOME/extraction_uwref/Sep_6229"
OUT_DIR="$HOME/autoencoder_outputs_${SLURM_JOB_ID:-manual}"
SCALER_DIR="$HOME/autoencoder_outputs_7503"

echo "[node] $(hostname)"
echo "[wd]   $(pwd)"

# Load anaconda3.
module load anaconda3/2024.10-py3.12.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
python -c "import sys; print('[python]', sys.executable, sys.version)"

mkdir -p "${OUT_DIR}"

echo "[info] Running Python..."
srun --ntasks=1 python run_ae2d_model.py \
  --data_dir "$IN_DIR" \
  --output_dir "$OUT_DIR" \
  --aug_data_dir "$SEC_IN_DIR" \
  --sep_data_dir "$TER_IN_DIR" \
  --scaler_dir "$SCALER_DIR" \
  --n_epochs 40 \
  --batch_size 256 \
  --segment_sec 5.0 \
  --T_target 300

echo "[info] Outputs saved to ${OUT_DIR}"