import torch
import os
import sys
import argparse
import json
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../steering"))
from local_replacement_models import LocalCLTReplacementModel
sys.path.append(os.path.join(os.path.dirname(__file__), "../training"))
from clt_module import CLTLightningModule

parser = argparse.ArgumentParser(description="Get edge weights from CLT model")
parser.add_argument("--base_folder", type=str, required=True, help="Base folder path")
args = parser.parse_args()

BATCH_SIZE = 32
EPSILON = 1e-10
CLT_CKPT = "../models/CLT_L6_D3200/checkpoints/last.ckpt"
ESM_PATH = "../models/esm2_t6_8M_UR50D.pt"


def print_edge_stats(edges_list, label):
    """
    edges_list rows are:
      [src_token, src_layer, src_latent, tgt_token, tgt_layer, tgt_latent, weight_signed]
    Stats are computed over abs(weight_signed).
    """
    print(f"\n{'='*50}")
    print(f"{label}: {len(edges_list)} edges with |weight| > {EPSILON}")
    if edges_list:
        mags = [abs(row[6]) for row in edges_list]
        print(f"  Avg |weight|: {sum(mags) / len(mags):.6f}")
        print(f"  Min |weight|: {min(mags):.6f}")
        print(f"  Max |weight|: {max(mags):.6f}")
    print(f"{'='*50}\n")


# Load inputs
activation_indices = json.load(open(os.path.join(args.base_folder, "activation_indices.json")))
seq = open(os.path.join(args.base_folder, "seq.txt")).read().strip()
print(f"Loaded {len(activation_indices)} activation indices, sequence length: {len(seq)}")

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

clt_pl = CLTLightningModule.load_from_checkpoint(
    CLT_CKPT, map_location=device, esm2_weight=ESM_PATH, strict=False
)
clt_pl.to(device).eval()
clt_local = LocalCLTReplacementModel(clt_pl, device, base_prompt=seq)
clt_local.compute_base_values()

# Parse: [layer, token, weight, feature_idx] -> (layer, token, feature_idx)
activations = [(a[0], a[1], a[3]) for a in activation_indices]

# Group by layer
activations_by_layer = defaultdict(list)
for layer, token, feature in activations:
    activations_by_layer[layer].append((layer, token, feature))

sorted_layers = sorted(activations_by_layer.keys(), reverse=True)

# Print node counts per layer
print(f"\nNodes per layer:")
for layer in sorted(activations_by_layer.keys()):
    print(f"  Layer {layer}: {len(activations_by_layer[layer])} nodes")
print(f"  Total: {len(activations)} nodes\n")

# Will store rows as:
# [src_token, src_layer, src_latent, tgt_token, tgt_layer, tgt_latent, weight_signed]
edges = []

print(f"Computing edge weights (epsilon={EPSILON}, batch_size={BATCH_SIZE})")

for tgt_layer in tqdm(sorted_layers, desc="Target layers"):
    sources = []
    for src_layer in range(tgt_layer):
        sources.extend(activations_by_layer[src_layer])

    if not sources:
        continue

    for tgt_layer, tgt_token, tgt_feature in tqdm(
        activations_by_layer[tgt_layer], desc=f"Layer {tgt_layer}", leave=False
    ):
        target = (tgt_layer, tgt_token, tgt_feature)

        for i in range(0, len(sources), BATCH_SIZE):
            source_batch = sources[i : i + BATCH_SIZE]
            weights = clt_local.compute_target_batched(source_batch, target, rerun_base=False)

            for (src_layer, src_token, src_feature), weight in zip(source_batch, weights):
                # keep signed weight, but threshold/filter by absolute magnitude
                w = float(weight)  # assumes weight is already a python float; ok if it's a 0-d tensor too
                if abs(w) > EPSILON:
                    edges.append([
                        int(src_token),
                        int(src_layer),
                        int(src_feature),
                        int(tgt_token),
                        int(tgt_layer),
                        int(tgt_feature),
                        w,  # signed
                    ])

print_edge_stats(edges, "All edges")


output_path = os.path.join(args.base_folder, "virtual_weights.json")

with open(output_path, "w") as f:
    json.dump(edges, f)

print(f"Saved to {output_path}")
