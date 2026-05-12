import torch
import os
import sys
import json
import argparse
import numpy as np

# --- Adjust Paths to match your environment ---
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../training"))
sys.path.append(os.path.abspath("../circuit_utils"))
sys.path.append(os.path.abspath("../steering"))

# Import CLT Module
try:
    from clt_module import CLTLightningModule
except ImportError:
    print("Warning: CLTLightningModule import failed. Ensure 'training' folder is in path.")
    CLTLightningModule = None

from local_replacement_models import LocalCLTReplacementModel

# ============================================================================
# Configuration
# ============================================================================
TOP_K = 10  # Number of top activations per layer

CLT_CKPT = "../models/CLT_L6_D3200/checkpoints/last.ckpt"
ESM_PATH = "../models/esm2_t6_8M_UR50D.pt"

# ============================================================================
# Parse Arguments
# ============================================================================
parser = argparse.ArgumentParser(description="Generate circuit from top activations per layer.")
parser.add_argument("--sequence", type=str, required=True, help="Protein sequence to analyze")
parser.add_argument("--output", type=str, required=True, help="Output path for circuit JSON")
args = parser.parse_args()

# ============================================================================
# Main Logic
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load model - latents_cache is populated automatically via compute_base_values
print("Loading model and computing latents...")
pl_module = CLTLightningModule.load_from_checkpoint(CLT_CKPT, esm2_weight=ESM_PATH, strict=False, weights_only=False)
pl_module.to(device).eval()
replacement_model = LocalCLTReplacementModel(pl_module, device, base_prompt=args.sequence)

# Extract top-k latent indices per layer from latents_cache
nodes = {}
for layer_idx, latents_T1D in replacement_model.latents_cache.items():
    latents_TD = latents_T1D.squeeze(1)  # (T, D)
    
    max_acts_per_latent_D, _ = latents_TD.max(dim=0)  # (D,)
    
    top_k_indices = torch.topk(max_acts_per_latent_D, k=min(TOP_K, max_acts_per_latent_D.shape[0])).indices

    # Convert to list of ints
    nodes[str(layer_idx)] = top_k_indices.cpu().tolist()
    
    print(f"Layer {layer_idx}: top {TOP_K} latents = {nodes[str(layer_idx)]}")

# Save circuit JSON
circuit = {"nodes": nodes}
with open(args.output, "w") as f:
    json.dump(circuit, f, indent=2)

print(f"Saved circuit to {args.output}")
