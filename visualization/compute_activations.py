import os
import sys
import torch
import polars as pl
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path so we can import family_circuit modules
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../training"))
from clt_module import CLTLightningModule

# --- Configuration ---
CLT_CHECKPOINT = os.environ.get("CLT_CHECKPOINT", "../models/CLT_L12_D4800/checkpoints/last.ckpt") 
PARQUET_PATH = os.environ.get("PARQUET_PATH", "../data/swissprot_full.parquet")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "visualizations")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))

class TopKActivationsTracker:
    """
    Tracks the Top-K activated sequences for every latent at every layer.
    Stores metadata + activations.
    """
    def __init__(self, num_layers, hidden_dim, k_sequences=10):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.k = k_sequences
        # storage[layer][latent_idx] = [ {score, metadata, trace}, ... ]
        self.storage = [[[] for _ in range(hidden_dim)] for _ in range(num_layers)]
        self.min_scores = torch.zeros((num_layers, hidden_dim)) - float('inf')

    def update(self, batch_activations_list, batch_max_scores, batch_metadata):
        """
        Args:
            batch_activations_list: List of L tensors, each (B, T, D). Full traces.
            batch_max_scores: Tensor (B, L, D). Max activation over time (for sorting).
            batch_metadata: List of dicts [{"Entry Name":..., "Sequence":...}, ...]
        """
        # Move scores to CPU for decision logic
        batch_max_scores = batch_max_scores.cpu()
        B, num_layers, hidden = batch_max_scores.shape
        
        for b in range(B):
            meta = batch_metadata[b]
            
            # Iterate Layers
            for l in range(num_layers):
                # optimization: pre-calculate which latents are candidates
                # A sequence is a candidate if its Max Score > Current K-th Best Score
                relevant_latents = torch.where(batch_max_scores[b, l] > self.min_scores[l])[0]
                
                if len(relevant_latents) == 0:
                    continue

                # shape: (T, D)
                # Slicing [b] keeps it on device until .cpu() is called
                layer_acts_TD = batch_activations_list[l][b]

                for lat_idx in relevant_latents:
                    lat_idx = lat_idx.item()
                    score = batch_max_scores[b, l, lat_idx].item()

                    # 1. Get the raw padded trace
                    raw_activations = layer_acts_TD[:, lat_idx].detach().cpu().numpy().astype(np.float32)

                    # 2. Calculate the valid length
                    # ESM adds <CLS> at start and <EOS> at end, so we want (1 + SeqLen + 1)
                    seq_len = len(meta["Sequence"])
                    valid_len = seq_len + 2

                    # 3. Slice to remove padding
                    # Safety check: ensure we don't slice beyond the tensor (in case logic differs)
                    valid_len = min(valid_len, len(raw_activations))
                    activations = raw_activations[:valid_len]

                    # Create Entry
                    entry = {
                        "Score": score,           # Max activation value (sorting key)
                        "Activations": activations, # Sequence activations
                        **meta                    # Unpack metadata (Entry Name, Sequence, etc.)
                    }
                    
                    current_list = self.storage[l][lat_idx]
                    current_list.append(entry)
                    current_list.sort(key=lambda x: x["Score"], reverse=True)
                    
                    # Trim to Top K
                    if len(current_list) > self.k:
                        current_list = current_list[:self.k]
                        self.storage[l][lat_idx] = current_list # <--- CRITICAL FIX
                    
                    # Update Threshold
                    if len(current_list) == self.k:
                        self.min_scores[l, lat_idx] = current_list[-1]["Score"]

    def save(self, path):
        print(f"Saving aggregated activations to {path}...")
        torch.save({
            "storage": self.storage, 
            "metadata": {"layers": self.num_layers, "hidden": self.hidden_dim, "k": self.k}
        }, path)

def get_clt_traces(model, tokens_BT):
    """
    Run the CLT forward pass.
    Returns:
        full_acts_list: List of [ (B, T, D) ] per layer.
        max_acts_BLD: (B, L, D) Tensor of max-over-time scores.
    """
    # 1. Collect Stack (B, T, Layers, H) -> (S, L, H)
    x_stack_flat_SLH, _, mask_S = model.collector.collect(tokens_BT)
    
    # 2. Reshape back to (B, T, L, H)
    B, T = tokens_BT.shape
    L = model.num_layers
    H = model.args.d_model
    D = model.args.d_hidden
    
    x_stack_BTLH = x_stack_flat_SLH.view(B, T, L, H)
    
    full_acts_list = []
    max_acts_BLD = torch.zeros((B, L, D), device=tokens_BT.device)
    
    # Process Layer-by-Layer to keep peak memory lower
    for l in range(L):
        # Input: (B, T, H)
        x_layer_BTH = x_stack_BTLH[:, :, l, :]
        
        # CLT Logic
        x_ln_BTH, _, _ = model.clt.LN(x_layer_BTH)
        x_ln_BTH = x_ln_BTH - model.clt.b_pre[l]
        
        pre_acts_BTD = model.clt.encoders[l](x_ln_BTH) + model.clt.b_enc[l]
        latents_BTD = model.clt.topK_activation(pre_acts_BTD, k=model.clt.k)
        
        # 1. Save Full Trace (B, T, D) for this layer
        # We assume sequence length T fits in memory. 
        full_acts_list.append(latents_BTD)
        
        # 2. Calculate Max Score (B, D)
        max_acts_BLD[:, l, :], _ = latents_BTD.max(dim=1)
        
    return full_acts_list, max_acts_BLD

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top_k_seqs", type=int, default=10)
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Data with specific columns
    print(f"Loading data from {PARQUET_PATH}...")
    # We select the requested columns. Ensure exact casing matches your parquet schema.
    needed_cols = ["Entry Name", "Protein names", "Entry", "Sequence"]
    df = pl.read_parquet(PARQUET_PATH).select(needed_cols)
    MAX_LEN = 1022
    df = df.with_columns(
        pl.col("Sequence").str.len_chars().alias("seq_len")
    ).filter(
        pl.col("seq_len") <= MAX_LEN
    )
    
    if args.limit:
        df = df.head(args.limit)
    
    # Convert to list of dicts for easy batching
    all_data = df.to_dicts()
    print(f"Processing {len(all_data)} sequences...")

    # 2. Load CLT Model
    print(f"Loading CLT from {CLT_CHECKPOINT}...")
    model = CLTLightningModule.load_from_checkpoint(CLT_CHECKPOINT, strict=False, weights_only=False)
    model.to(device)
    model.eval()
    
    # 3. Initialize Tracker
    tracker = TopKActivationsTracker(
        model.args.num_layers, 
        model.args.d_hidden, 
        k_sequences=args.top_k_seqs
    )

    # 4. Process Loop
    for i in tqdm(range(0, len(all_data), BATCH_SIZE)):
        batch_items = all_data[i : i + BATCH_SIZE]
        
        # Extract sequences for tokenization
        batch_seqs = [item["Sequence"] for item in batch_items]
        
        with torch.no_grad():
            tokens = model.tokenize(batch_seqs)
            
            # Get Activations
            # full_acts_list: List of L tensors (B, T, D)
            # max_acts_BLD: Tensor (B, L, D)
            full_acts_list, max_acts_BLD = get_clt_traces(model, tokens)
            
            # Update Tracker with Metadata
            tracker.update(full_acts_list, max_acts_BLD, batch_items)

    # 5. Save
    save_path = "top10_activations_35M.pt"
    tracker.save(save_path)
    print("Done!")

if __name__ == "__main__":
    main()