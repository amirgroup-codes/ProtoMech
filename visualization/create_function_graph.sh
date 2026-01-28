#!/bin/bash
# ==============================================================================
# Graph Pipeline: Analyze sequence, compute edge weights, and generate graph
# ==============================================================================
NAME="GB1"
# circuit_json OPTIONAL - leave empty to auto-generate from SEQUENCE1
CIRCUIT_JSON="../function_circuit/functions/CLT_sequential/multiples/SPG1_STRSG_Olson_2014/rand_multiples_fold0.json"
SEQUENCE1="QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
SEQUENCE2="QYKLILNGKTLKGETTTEAVDAWTAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
SEQUENCE3="QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE" #optional, currently set as WT
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "=============================================="
echo "Generating For: $NAME"
echo "=============================================="

# Step 0: Generate circuit JSON if not provided
if [ -z "$CIRCUIT_JSON" ] || [ ! -f "$CIRCUIT_JSON" ]; then
    echo "[0/3] No circuit JSON provided, generating from sequence..."
    mkdir -p "./$NAME"
    CIRCUIT_JSON="./${NAME}/${NAME}_circuit.json"
    
    python circuit_top_acts.py \
        --sequence "$SEQUENCE1" \
        --output "$CIRCUIT_JSON"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate circuit JSON"
        exit 1
    fi
    echo "Generated circuit JSON: $CIRCUIT_JSON"
fi

# Step 1: Analyze sequences
echo "[1/2] Analyzing sequences..."
if [ -n "$SEQUENCE3" ]; then
    python circuit_analysis_function.py \
        --sequence1 "$SEQUENCE1" \
        --sequence2 "$SEQUENCE2" \
        --sequence3 "$SEQUENCE3" \
        --circuit_json "$CIRCUIT_JSON" \
        --entry_name "$NAME"
else
    python circuit_analysis_function.py \
        --sequence1 "$SEQUENCE1" \
        --sequence2 "$SEQUENCE2" \
        --circuit_json "$CIRCUIT_JSON" \
        --entry_name "$NAME"
fi
# Step 2: Compute edge weights (for seq1 as base)
echo "[2/2] Computing edge weights..."
python get_edge_weights.py --base_folder "./$NAME/seq1"
python get_edge_weights.py --base_folder "./$NAME/seq2"
if [ -n "$SEQUENCE3" ] && [ -d "./$NAME/seq3" ]; then
    python get_edge_weights.py --base_folder "./$NAME/seq3"
fi
echo ""
echo "Done! Proceed to generate the graph on the website."