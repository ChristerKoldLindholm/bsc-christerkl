#!/usr/bin/env bash
#SBATCH --job-name=pc_kmeans_experiment
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=30
#SBATCH --mem=35G
#SBATCH -p gpu --gres=gpu:titanrtx:2

set -eo pipefail

# CONFIGS
SID="4121" # Downsampled feature set folder. 
CONDA_ENV="venv_stream_py312"
IN_DIR="$HOME/clustering_outputs_${SID}"
OUT_DIR="$HOME/clustering_outputs_${SLURM_JOB_ID:-manual}"

echo "[node] $(hostname)"
echo "[wd]   $(pwd)"

# Load anaconda3.
module load anaconda3/2024.10-py3.12.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
python -c "import sys; print('[python]', sys.executable, sys.version)"

mkdir -p "${OUT_DIR}"

echo "[info] Running clustering..."
srun --ntasks=1 python run_components_kmeans_experiment_sample.py \
  --input-root "$IN_DIR" \
  --saved-pca "$IN_DIR/pca_kmeans_batches.npz" \
  --output-root "$OUT_DIR" \
  --pc-list 2 3 4 5 6 7 8 16 32 64 128 256 \
  --batch-files 512 --n-mels 128 --drop-mels 40 --topn 10 \
  --downsampled-dir "$IN_DIR/downsampled_features" \
  --save-plotting \
  --sample-files 2000 \
  --sample-seed 104

echo "[info] Clustering outputs saved to ${OUT_DIR}"