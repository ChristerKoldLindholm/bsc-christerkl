#!/usr/bin/env bash
#SBATCH --job-name=pca_clust_elbow
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH -p gpu --gres=gpu:titanrtx:2

set -eo pipefail

# CONFIGS
SID="9979" # Feature set folder. 
CONDA_ENV="venv_stream_py312"
IN_DIR="$HOME/logmel_outputs_${SID}"
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
srun --ntasks=1 python run_pca_cluster_elbow.py \
  --input-root "$IN_DIR" \
  --output-root "$OUT_DIR"

echo "[info] Clustering outputs saved to ${OUT_DIR}"