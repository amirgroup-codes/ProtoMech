#!/bin/bash
# ==============================================================================
# Graph Pipeline: Analyze sequence, compute edge weights, and generate graph
# ==============================================================================
NAME="kinase"
# circuit_json OPTIONAL - leave empty to auto-generate from SEQUENCE
# CIRCUIT_JSON="../family_circuit/families/CLT_sequential/IPR000719.json"
SEQUENCE="MVKQVDFAEVKLSEKFLGAGSGGAVRKATFQNQEIAVKIFDFLEETIKKNAEREITHLSEIDHENVIRVIGRASNGKKDYLLMEYLEEGSLHNYLYGDDKWEYTVEQAVRWALQCAKALAYLHSLDRPIVHRDIKPQNMLLYNQHEDLKICDFGLATDMSNNKTDMQGTLRYMAPEAIKHLKYTAKCDVYSFGIMLWELMTRQLPYSHLENPNSQYAIMKAISSGEKLPMEAVRSDCPEGIKQLMECCMDINPEKRPSMKEIEKFLGEQYESGTDEDFIKPLDEDTVAVVTYHVDSSGSRIMRVDFWRHQLPSIRMTFPIVKREAERLGKTVVREMAKAAADGDREVRRAEKDTERETSRAAHNGERETRRAGQDVGRETVRAVKKIGKKLRF"
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# echo the script directory:
echo "Script directory: $SCRIPT_DIR"
echo "=============================================="
echo "Generating For: $NAME"
echo "=============================================="

# Step 0: Generate circuit JSON if not provided
if [ -z "$CIRCUIT_JSON" ] || [ ! -f "$CIRCUIT_JSON" ]; then
    echo "[0/2] No circuit JSON provided, generating from sequence..."
    mkdir -p "./$NAME"
    CIRCUIT_JSON="./${NAME}/${NAME}_circuit.json"
    
    python circuit_top_acts.py \
        --sequence "$SEQUENCE" \
        --output "$CIRCUIT_JSON"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate circuit JSON"
        exit 1
    fi
    echo "Generated circuit JSON: $CIRCUIT_JSON"
fi

# Step 1: Analyze sequence
echo "[1/2] Analyzing sequence..."
python circuit_analysis.py \
    --sequence "$SEQUENCE" \
    --circuit_json "$CIRCUIT_JSON" \
    --entry_name "$NAME"
# Step 2: Compute edge weights
echo "[2/2] Computing edge weights..."
python get_edge_weights.py --base_folder "./$NAME"
echo ""
echo "Done! Proceed to generate the graph on the website."