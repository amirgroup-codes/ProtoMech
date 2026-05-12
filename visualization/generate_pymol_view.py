import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import argparse
import numpy as np
from Bio import PDB
from pathlib import Path

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# 1. Local PDB Path
PDB_NAME = '1PGA.cif' 

# 2. Sequence (Must match PDB numbering 1-to-1)
SEQUENCE = "QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

# 3. Targets: {Layer: [Latents]}
TARGET_NODES = {
    0: [248],
    2: [3026], # Completely killed on W in low fitness
    5: [3028, 1609], # High fitness activates now!
}

# 4. Analysis Window (1-indexed PDB numbering)
START_POS = 2
END_POS = 56
PDB_OFFSET = 2

# =============================================================================
# IMPORTS & MODEL SETUP
# =============================================================================

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../training"))
sys.path.append(os.path.abspath("../circuit_utils"))
sys.path.append(os.path.abspath("../steering"))

try:
    from esm_activation import ESM2ActivationCollector
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_utils"))
        from esm_activation import ESM2ActivationCollector
    except ImportError:
        ESM2ActivationCollector = None
        print("Warning: ESM2ActivationCollector not found.")

try:
    from clt_module import CLTLightningModule
except ImportError:
    print("Error: Could not import CLTLightningModule.")
    sys.exit(1)

from local_replacement_models import LocalCLTReplacementModel

# Define Model Wrapper Inline to ensure standalone execution
class FullReplacementModel(nn.Module):
    def __init__(self, pl_module, device):
        super().__init__()
        self.pl_module = pl_module
        self.device = device
        self.model = pl_module.clt if hasattr(pl_module, 'clt') else pl_module.plt
        self.esm = pl_module.esm_model
        self.alphabet = pl_module.alphabet
        self.num_layers = self.model.num_layers

        if ESM2ActivationCollector is not None:
            self.collector = ESM2ActivationCollector(self.esm, self.alphabet)
            self.collector.register_hooks()

    def tokenize(self, seqs):
        return self.pl_module.tokenize(seqs)

    def _encode_latents(self, l, x_mlp_in_TBH):
        x_norm_TBH, mu, std = self.model.LN(x_mlp_in_TBH)
        x_norm_TBH = x_norm_TBH - self.model.b_pre[l]
        enc_TBD = self.model.encoders[l](x_norm_TBH) + self.model.b_enc[l]
        latents_TBD = self.model.topK_activation(enc_TBD, k=self.model.k)
        return latents_TBD, enc_TBD, mu, std

    def _decode_latents(self, l, current_latents_list):
        T, B, D = current_latents_list[l].shape
        recon_TBH = torch.zeros(T, B, self.esm.embed_dim, device=self.device)
        for src in range(l + 1):
            key = f"{src}_{l}"
            if key in self.model.decoders:
                recon_TBH = recon_TBH + (current_latents_list[src] @ self.model.decoders[key])
        return recon_TBH

    def layer_forward(self, l, x_prev_TBH, latents_list_L, x_gt=None, ablate_nodes=None, padding_mask=None):
        layer = self.esm.layers[l]
        residual = x_prev_TBH
        if x_gt is not None:
            x_ln = layer.self_attn_layer_norm(x_gt)
        else:
            x_ln = layer.self_attn_layer_norm(x_prev_TBH)

        x_attn_out, _ = layer.self_attn(
            query=x_ln, key=x_ln, value=x_ln,
            key_padding_mask=padding_mask, need_weights=False
        )
        x_TBH = residual + x_attn_out

        residual = x_TBH
        x_mlp_in_TBH = layer.final_layer_norm(x_TBH)

        with torch.no_grad():
            gt_mlp_TBH = layer.fc2(F.gelu(layer.fc1(x_mlp_in_TBH)))

        latents_TBD, enc_TBD, mu, std = self._encode_latents(l, x_mlp_in_TBH)
        
        current_latents_list = latents_list_L + [latents_TBD]
        recon_TBH = self._decode_latents(l, current_latents_list)
        recon_TBH = recon_TBH + self.model.b_pre[l]
        recon_TBH = recon_TBH * std + mu

        x_curr_TBH = residual + recon_TBH
        gt_mlp_TBH = residual + gt_mlp_TBH

        return x_curr_TBH, latents_TBD, recon_TBH, gt_mlp_TBH

    def forward(self, batch_seqs, freeze_attention=False):
        self.model.eval()
        latents_list_L = []
        
        tokens_BT = self.tokenize(batch_seqs).to(self.device)
        B, T = tokens_BT.shape
        H = self.esm.embed_dim

        with torch.no_grad():
            x_curr_BTH = self.esm.embed_scale * self.esm.embed_tokens(tokens_BT)

        padding_mask = (tokens_BT == self.alphabet.padding_idx)
        x_curr_TBH = x_curr_BTH.transpose(0, 1)

        x_stack_SLH = None
        if freeze_attention:
            if not hasattr(self, 'collector') or self.collector is None:
                pass 
            else:
                x_stack_SLH, _, _, _, _ = self.collector.collect(tokens_BT)

        for l in range(self.num_layers):
            x_gt = x_stack_SLH[:, l, :].view(B, T, H).transpose(0, 1) if x_stack_SLH is not None else None

            x_curr_TBH, latents_TBD, recon_TBH, gt_mlp_TBH = self.layer_forward(
                l, x_curr_TBH, latents_list_L, x_gt=x_gt, padding_mask=padding_mask
            )
            latents_list_L.append(latents_TBD)

        return latents_list_L

# -----------------------------------------------------------------------------
# 2. Parsing & Mapping
# -----------------------------------------------------------------------------

def parse_pdb_residues_local(filepath, chain_id="A"):
    if filepath.endswith(".cif"):
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure("model", filepath)
        model = next(iter(structure))
    except Exception as e:
        print(f"Error parsing PDB: {e}")
        return [], None

    if chain_id not in model:
        chains = list(model.get_chains())
        if not chains: return [], None
        chain = chains[0]
        chain_id = chain.id
        print(f"Note: Using chain {chain_id}")
    else:
        chain = model[chain_id]
        
    residue_ids = []
    for res in chain:
        if res.id[0] == " ":
            residue_ids.append(res.id[1])
            
    return residue_ids, chain_id

def get_activation_map(full_trace, pdb_ids, start_pos, end_pos):
    """
    Maps Sequence Trace -> PDB Residue IDs.
    
    Logic:
    - trace[0] is [CLS] (dummy).
    - trace[1] is the 1st residue of SEQUENCE.
    - The 1st residue of SEQUENCE corresponds to PDB Residue # PDB_OFFSET.
    
    Therefore:
      PDB_Res_Num maps to Trace_Index = (PDB_Res_Num - PDB_OFFSET) + 1
    """
    mapping = {}
    
    for res_num in pdb_ids:
        # Trace Index = PDB Residue Num (assuming 1-based indexing)
        trace_idx = (res_num - PDB_OFFSET) + 1
        
        if trace_idx < len(full_trace):
            val = float(full_trace[trace_idx])
            
            # Apply Window Filter
            if start_pos is not None and res_num < start_pos:
                val = 0.0
            if end_pos is not None and res_num > end_pos:
                val = 0.0
                
            mapping[res_num] = val
    return mapping

# -----------------------------------------------------------------------------
# 3. Generate Python Script for PyMOL
# -----------------------------------------------------------------------------

def generate_combined_py(pdb_path, chain_id, all_activations, output_dir="pymol_viz"):
    filename = "circuit_visualization.py"
    filepath = os.path.join(output_dir, filename)
    
    lines = [
        "from pymol import cmd",
        "",
        "cmd.reinitialize()",
        "cmd.bg_color('white')",
        "cmd.set('ray_trace_mode', 1)",
        "cmd.set('ray_shadows', 0)",
        "cmd.set('antialias', 2)",
        "",
        "# Custom Colors",
        "cmd.set_color('base_blue', [91/255, 150/255, 210/255])",
        "",
        "# Load Structure",
        f"cmd.load('{pdb_path}', 'base_struct')",
        "cmd.hide('everything', 'base_struct')",
        "cmd.color('base_blue', 'base_struct')",
        "cmd.show('cartoon', 'base_struct')",
        "",
        "# Coloring Helper (Normalized)",
        "def apply_spectrum_norm(obj_name, raw_max):",
        "    cmd.color('base_blue', obj_name)",
        "    print(f'Object {obj_name}: Raw Max = {raw_max:.4f}')",
        "    if raw_max < 0.0001: return",
        "    # Coloring only residues > 0.1 (10% max activation)",
        "    selection = f'{obj_name} and b > 0.1'",
        "    # White -> Red Spectrum",
        "    cmd.spectrum('b', 'white_red', selection=selection, minimum=0.1, maximum=1.0)",
        ""
    ]
    
    for (layer, latent), act_map in all_activations.items():
        obj_name = f"L{layer}_{latent}"
        vals = list(act_map.values())
        raw_max = max(vals) if vals else 0.0
        
        # --- NORMALIZATION ---
        norm_factor = 1.0 / raw_max if raw_max > 0 else 0.0
        
        lines.append(f"# --- {obj_name} (Raw Max: {raw_max:.4f}) ---")
        lines.append(f"cmd.create('{obj_name}', 'base_struct')")
        lines.append(f"cmd.alter('{obj_name}', 'b=0.0')")
        
        # Efficient writing
        for r, v in act_map.items():
            if v > 0.001:
                norm_val = v * norm_factor
                lines.append(f"cmd.alter('{obj_name} and chain {chain_id} and resi {r}', 'b={norm_val:.4f}')")
        
        lines.append(f"apply_spectrum_norm('{obj_name}', {raw_max})")
        lines.append(f"cmd.group('Circuit_Analysis', '{obj_name}')")
        lines.append("")

    lines += [
        "cmd.disable('base_struct')",
        "cmd.disable('Circuit_Analysis')", 
        "cmd.zoom('base_struct')",
        "print('Done! Enable specific objects in Circuit_Analysis to view.')"
    ]

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    return filepath

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clt_ckpt", type=str, default="../models/CLT_L6_D3200/checkpoints/last.ckpt")
    parser.add_argument("--esm_path", type=str, default="../models/esm2_t6_8M_UR50D.pt")
    parser.add_argument("--output_dir", type=str, default="pymol_viz")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print(f"Parsing PDB: {PDB_NAME}...")
    pdb_ids, chain_id = parse_pdb_residues_local(PDB_NAME)
    if not pdb_ids:
        print("Error: PDB parsing failed.")
        return

    print("Loading Model...")
    pl_module = CLTLightningModule.load_from_checkpoint(args.clt_ckpt, esm2_weight=args.esm_path, strict=False, weights_only=False)
    pl_module.to(device).eval()
    #model = FullReplacementModel(pl_module, device)
    model = LocalCLTReplacementModel(pl_module, device, base_prompt=SEQUENCE)
    
    print(f"Running inference on Sequence ({len(SEQUENCE)} AA)...")
    latents_list = model.latents_cache
    # with torch.no_grad():
    #     latents_list = model.forward([SEQUENCE], freeze_attention=True)

    all_activations = {}
    for layer, latents in TARGET_NODES.items():
        for latent in latents:
            print(f"Processing L{layer}-{latent}...")
            tensor = latents_list[layer]
            
            # --- Robust Shape Handling ---
            # Check if dim 1 matches seq length (Batch, Time, Dim)
            if tensor.shape[1] == len(SEQUENCE):
                trace = tensor.squeeze(0)[:, latent].cpu().numpy()
            else:
                trace = tensor.squeeze(1)[:, latent].cpu().numpy()
                
            act_map = get_activation_map(trace, pdb_ids, START_POS, END_POS)
            all_activations[(layer, latent)] = act_map

    Path(args.output_dir).mkdir(exist_ok=True)
    out_path = generate_combined_py(PDB_NAME, chain_id, all_activations, args.output_dir)
    
    print("-" * 40)
    print(f"✅ Generated Python script: {out_path}")
    print(f"Run in PyMOL via 'File -> Run Script...'")
    print("-" * 40)

if __name__ == "__main__":
    main()