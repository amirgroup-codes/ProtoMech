#!/bin/bash
set -e 
# Usage:
#   sh main.sh
#   sh main.sh --overwrite


REPO_ROOT="$(dirname "$(pwd)")"
DMS_DATA_DIR="$REPO_ROOT/function_circuit/DMS" 

# ESM parameters
LAYERS=6
HIDDEN_SIZE=320

# Hyperparameters
export BATCH_SIZE=16

# Model Checkpoints
export CLT_CHECKPOINT="../models/CLT_L6_D3200/checkpoints/last.ckpt"
export PLT_CHECKPOINT="../models/PLT_L6_D3200/checkpoints/last.ckpt" 
export ESM_WEIGHTS="../models/esm2_t6_8M_UR50D.pt"

echo "========================================"
echo " [Setup] Configuration"
echo "========================================"
echo "  > ESM Model:        ${LAYERS} Layers, ${HIDDEN_SIZE} Dim"
echo "  > DMS Data Dir:       $DMS_DATA_DIR"

ARGS="$@"

python 01_discover_circuits.py \
    --dms_root "DMS" \

echo "Pipeline Complete."