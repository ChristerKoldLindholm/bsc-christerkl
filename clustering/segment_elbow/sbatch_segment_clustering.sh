#!/usr/bin/env bash
#SBATCH --job-name=seg_elbow
#SBATCH --ntasks=1
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=30
#SBATCH --mem=35G
#SBATCH -p gpu --gres=gpu:titanrtx:2

set -eo pipefail

# CONFIGS
PROGRAM="run_segment_clustering.py"
CONDA_ENV="venv_stream_py311"
IN_DIR="$HOME/second_extraction/"
OUT_DIR="$HOME/seg_clust_${SLURM_JOB_ID:-manual}"

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
  --max-pc 8192 \
  --min_i-pc 5 \
  --pc_list 4096 6144 8192 \
  --batch-segments 13000 \
  --segment-sec 5.0 \
  --clip-sec 270.0 \
  --n-mels 128 \
  --mels-start 9 \
  --mels-end 60 \
  --downsample-target 300 \
  --topn 500 \
  --ks-list 2 4 6 8 10 12 14 16 20 24 32 \
  --sample-files 2000 \
  --sample-seed 104

echo "[info] Clustering outputs saved to ${OUT_DIR}"