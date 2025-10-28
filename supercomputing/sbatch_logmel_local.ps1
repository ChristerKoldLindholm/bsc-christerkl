#!/usr/bin/env pwsh
#ErrorActionPreference = "Stop"

# CONFIGS
$IN_DIR="D:\data_dryad"
$OUT_DIR="C:\Users\Lindholm\Documents\BSc\bsc_project\supercomputing/testing"
$SUBSET_LEN=10

# Run the Python driver.
Write-Host "[info] Running log-mel feature extraction."
py run_logmel_slurm.py `
  --input-root "$IN_DIR" `
  --output-root "$OUT_DIR" `
  --subset-len "$SUBSET_LEN"

Write-Host "[done] Features written to: $OUT_DIR"
