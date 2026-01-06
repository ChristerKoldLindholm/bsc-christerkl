#!/usr/bin/env bash
#SBATCH --job-name=pca_kmeans_full_segs
#SBATCH --ntasks=1
#SBATCH --time=35:00:00
#SBATCH --cpus-per-task=18
#SBATCH --mem=28G
#SBATCH -p gpu --gres=gpu:titanrtx:1

set -eo pipefail

# CONFIGS
PROGRAM="run_pca_kmeans_full_set_segs.py"
CONDA_ENV="venv_stream_py311"
IN_DIR="$HOME/second_extraction/"
OUT_DIR="$HOME/segs_clust_${SLURM_JOB_ID:-manual}"

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
srun --ntasks=1 python "$PROGRAM" \
  --input-root "$IN_DIR" \
  --output-root "$OUT_DIR" \
  --batch-segments 13000 \
  --segment-sec 5.0 \
  --topn 500 \
  --n-components 8192 \
  --k 8 \
  --mels-start 9 \
  --mels-end 60 \
  --downsample-target 300 \
  --random-seed 104

echo "[info] Clustering outputs saved to ${OUT_DIR}"