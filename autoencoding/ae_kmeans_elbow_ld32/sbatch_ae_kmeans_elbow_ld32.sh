#!/usr/bin/env bash
#SBATCH --job-name=ae_kmeans_elbow_hd16
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=30
#SBATCH --mem=35G
#SBATCH -p gpu --gres=gpu:titanrtx:4

set -eo pipefail

# CONFIGS
CONDA_ENV="venv_stream_py311"
SID="4121" # Downsampled feature set folder. 
IN_ROOT="$HOME/clustering_outputs_${SID}"
FEAT_DIR="$IN_ROOT/downsampled_features"
OUT_DIR="$HOME/ae_outputs_${SLURM_JOB_ID:-manual}"

echo "[node] $(hostname)"
echo "[wd]   $(pwd)"

# Load anaconda3.
module load anaconda3/2024.10-py3.12.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
python -c "import sys; print('[python]', sys.executable, sys.version)"

mkdir -p "${OUT_DIR}"

echo "[info] Running autoencoder + k-means elbow method..."

srun --ntasks=1 python run_ae_kmeans_elbow_ld32.py \
  --input-root "$FEAT_DIR" \
  --saved-pca "$IN_ROOT/pca_kmeans_batches.npz" \
  --output-root "$OUT_DIR" \
  --batch-size 64 \
  --latent-dim 32 \
  --hidden-dim 32 \
  --max-epochs 100 \
  --patience 10 \
  --min-delta 1e-4 \
  --sample-files 2048 \
  --sample-seed 104 \
  --k-min 2 \
  --k-max 16 \
  --lr 1e-3 \
  --weight-decay 1e-5

echo "[info] AE outputs saved to ${OUT_DIR}"