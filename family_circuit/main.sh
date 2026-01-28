#!/bin/bash
set -e
# Usage:
#   Full run:
#     sh main.sh
#
#   Full run (limit 10 families, overwrite):
#     sh main.sh --limit 10 --overwrite
#
#   Target specific family:
#     sh main.sh --target IPR000724

REPO_ROOT="$(dirname "$(pwd)")"
TRAINING_DIR="$REPO_ROOT/training" 
OUTPUT_DIR="families"
MASTER_NPZ_NAME="all_acts.npz"

# ESM parameters
LAYERS=6
HIDDEN_SIZE=320

# Hyperparameters
export BATCH_SIZE=16
export MIN_POSITIVES=2

# Model Checkpoints
export CLT_CHECKPOINT="../models/CLT_L6_D3200/checkpoints/last.ckpt"
export PLT_CHECKPOINT="../models/PLT_L6_D3200/checkpoints/last.ckpt" 
export SAE_CHECKPOINT="../models/SAE_L5_H19200/checkpoints/loss=0.04.ckpt" 
export ESM_WEIGHTS="../models/esm2_t6_8M_UR50D.pt"
export PARQUET_PATH="../data/swissprot_seqid30_75k_all_info_with_3di.parquet"
export OUTPUT_DIR="$OUTPUT_DIR"
export MASTER_NPZ_NAME="$MASTER_NPZ_NAME"
export PYTHONPATH="$TRAINING_DIR:$PYTHONPATH"

echo "========================================"
echo " [Setup] Configuration"
echo "========================================"
echo "  > Training Modules: $TRAINING_DIR"
echo "  > ESM Model:        ${LAYERS} Layers, ${HIDDEN_SIZE} Dim"
echo "  > Output Dir:       $OUTPUT_DIR"

ARGS="$@"

echo ""
echo "========================================"
echo " [Step 1] Extracting ESM embeddings"
echo "========================================"
python 01_extract_embeddings.py \
    --layers $LAYERS \
    --hidden_size $HIDDEN_SIZE \
    $ARGS

echo ""
echo "========================================"
echo " [Step 2] Discovering Circuits (CLT and PLT)"
echo "========================================"
echo "Starting CLT Circuit Discovery..."
echo ">>> Running CLT Mode: SEQUENTIAL..."

python 02_discover_circuits_clt.py \
    --recovery_ratio 0.7 \
    --max_nodes 1000 \
    --sequential \
#     --no_freeze_attention \
    $ARGS

echo ">>> Running CLT Mode: DIRECT..."
python 02_discover_circuits_clt.py \
    --recovery_ratio 0.7 \
    --max_nodes 1000 \
    $ARGS

echo "Starting PLT Circuit Discovery..."
python 02_discover_circuits_plt.py \
    --recovery_ratio 0.7 \
    --max_nodes 1000 \
    --no_freeze_attention \
    $ARGS

python 02_discover_circuits_sae.py \
    --recovery_ratio 0.7 \
    --max_nodes 1000
echo "Pipeline Complete."