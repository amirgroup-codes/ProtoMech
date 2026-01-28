import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Configuration
TARGET_PER_LAYER_DIFF = 5   # Nodes per layer for Differential Report
TARGET_PER_LAYER_SINGLE = 5 # Nodes per layer for Single Sequence Reports

# Import ESM2ActivationCollector
try:
    from esm_activation import ESM2ActivationCollector
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_utils"))
        from esm_activation import ESM2ActivationCollector
    except ImportError:
        ESM2ActivationCollector = None
        print("Warning: ESM2ActivationCollector not found.")

# Import CLT Module
try:
    from clt_module import CLTLightningModule
except ImportError:
    print("Warning: CLTLightningModule import failed. Ensure 'training' folder is in path.")
    CLTLightningModule = None

#from full_replacement_models import FullCLTReplacementModel
from local_replacement_models import LocalCLTReplacementModel

# -----------------------------------------------------------------------------
# Analysis Helper Functions
# -----------------------------------------------------------------------------

def get_adaptive_motif_window(sequence, trace, peak_trace_idx, min_radius=10, buffer=3, top_k=None):
    """Extracts a sequence window centered on peak_trace_idx."""
    seq_idx = peak_trace_idx - 1
    seq_len = len(sequence)

    if seq_idx < 0: return "<CLS>", 0, 0
    if seq_idx >= seq_len: return "<EOS>", seq_len, seq_len

    scan_start_t = 1
    scan_end_t = len(trace)
    
    local_trace_slice = trace[scan_start_t:scan_end_t]
    if len(local_trace_slice) == 0:
        return "", 0, 0
        
    positive_indices_local = [i for i, val in enumerate(local_trace_slice) if val > 0]
    global_active_indices = [idx + scan_start_t for idx in positive_indices_local]
    
    if top_k is not None:
        global_active_indices.sort(key=lambda idx: trace[idx], reverse=True)
        important_indices = global_active_indices[:top_k]
    else:
        important_indices = global_active_indices

    important_indices.append(peak_trace_idx)
    min_important = min(important_indices)
    max_important = max(important_indices)

    start_trace_idx = min(peak_trace_idx - min_radius, min_important - buffer)
    end_trace_idx = max(peak_trace_idx + min_radius + 1, max_important + buffer + 1)
    
    start_seq = max(0, start_trace_idx - 1)
    end_seq = min(seq_len, end_trace_idx - 1)

    window_data = [] 
    if end_seq - start_seq <= 0: return "", 0, 0

    for pos in range(start_seq, end_seq):
        char = sequence[pos]
        val = float(trace[pos + 1])
        window_data.append({'pos': pos, 'char': char, 'val': val})

    highlight_indices = set()
    if top_k is not None:
        window_data_sorted = sorted(window_data, key=lambda x: x['val'], reverse=True)
        for item in window_data_sorted[:top_k]:
            if item['val'] > 0: highlight_indices.add(item['pos'])
    else:
        for item in window_data:
            if item['val'] > 0: highlight_indices.add(item['pos'])
    
    motif_str = ""
    for item in window_data:
        if item['pos'] in highlight_indices:
            motif_str += f"[{item['char']}]"
        else:
            motif_str += item['char']
            
    return motif_str, start_seq, end_seq

def format_global_hit(hit):
    entry = hit.get('Entry', '?')
    name = hit.get('Entry Name', '')
    pname = hit.get('Protein names', '')
    score = hit.get('Score', 0.0)
    
    seq = hit.get('Sequence') or hit.get('seq')
    if not seq:
        return f"{entry} ({name}) - Score: {score:.4f} (Sequence data unavailable)"

    trace = hit.get('Activations')
    if trace is not None and isinstance(trace, torch.Tensor):
        trace = trace.detach().cpu().numpy()

    peak_idx = hit.get('Peak_Index') or hit.get('peak_idx')
    if peak_idx is None:
        if trace is not None:
            peak_idx = np.argmax(trace) 
        else:
            return f"{entry} ({name}) - Score: {score:.4f} (Peak location unavailable)"

    center_seq_idx = peak_idx - 1
    start = max(0, center_seq_idx - 10)
    end = min(len(seq), center_seq_idx + 11)
    
    snippet = ""
    bracket_indices = set()
    
    if trace is not None:
        t_start = max(0, start + 1)
        t_end = min(len(trace), end + 1)
        if t_end > t_start:
            local_trace = trace[t_start:t_end]
            for i, val in enumerate(local_trace):
                if val > 0: bracket_indices.add(start + i)
    else:
        bracket_indices.add(center_seq_idx)
        
    for i in range(start, end):
        char = seq[i]
        if i in bracket_indices: snippet += f"[{char}]"
        else: snippet += char
            
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(seq) else ""
    return f"{entry} ({name}) - Score: {score:.4f} - {prefix}{snippet}{suffix} - {pname}"

def print_result_block(f, rank_label, item, sequence, ref_storage):
    """Standard Single-Sequence Report Block"""
    layer = item['layer']
    latent = item['latent']
    score = item['score']
    trace = item['trace']
    peak_idx = item['peak_idx']
    
    motif_str, seq_start, seq_end = get_adaptive_motif_window(
        sequence, trace, peak_idx, min_radius=5, buffer=3, top_k=None 
    )
    
    t_start = max(0, seq_start + 1)
    t_end = min(len(trace), seq_end + 1)
    trace_window = trace[t_start:t_end]
    
    f.write(f"--- {rank_label} ---\n")
    f.write(f"Node: Layer {layer}, Latent {latent}\n")
    f.write(f"Max Activation: {score:.4f}\n")
    f.write(f"Peak Location: Trace Idx {peak_idx} (AA #{peak_idx})\n")
    f.write(f"Motif Context: {motif_str}\n")
    f.write(f"Trace Window : {trace_window}\n")
    f.write("\n")
    
    f.write(f"   [Global Top 10 Reference Entries for L{layer}-{latent}]\n")
    if ref_storage and layer < len(ref_storage) and latent < len(ref_storage[layer]):
        top_hits = ref_storage[layer][latent]
        if top_hits:
            for i, hit in enumerate(top_hits[:10]):
                f.write(f"   {i+1}. {format_global_hit(hit)}\n")
        else:
            f.write("   (No activations recorded in dataset)\n")
    else:
        f.write("   (Reference not loaded or index out of bounds)\n")
    f.write("\n")

def print_differential_result_block(f, rank_label, item, seqA, seqB, ref_storage):
    """Differential Report Block showing both sequences side-by-side"""
    layer = item['layer']
    latent = item['latent']
    diff = item['diff']
    
    sA_data = item['seqA_data']
    sB_data = item['seqB_data']
    
    f.write(f"--- {rank_label} (Diff: {diff:.4f}) ---\n")
    f.write(f"Node: Layer {layer}, Latent {latent}\n")
    
    # Seq A Info
    sA_motif, sA_start, sA_end = get_adaptive_motif_window(seqA, sA_data['trace'], sA_data['peak_idx'], min_radius=5)
    tA_start = max(0, sA_start + 1)
    tA_end = min(len(sA_data['trace']), sA_end + 1)
    
    f.write(f"Seq1 Max: {sA_data['score']:.4f} @ AA #{sA_data['peak_idx']}\n")
    f.write(f"Seq1 Context: {sA_motif}\n")
    f.write(f"Seq1 Trace  : {sA_data['trace'][tA_start:tA_end]}\n\n")

    # Seq B Info
    sB_motif, sB_start, sB_end = get_adaptive_motif_window(seqB, sB_data['trace'], sB_data['peak_idx'], min_radius=5)
    tB_start = max(0, sB_start + 1)
    tB_end = min(len(sB_data['trace']), sB_end + 1)
    
    f.write(f"Seq2 Max: {sB_data['score']:.4f} @ AA #{sB_data['peak_idx']}\n")
    f.write(f"Seq2 Context: {sB_motif}\n")
    f.write(f"Seq2 Trace  : {sB_data['trace'][tB_start:tB_end]}\n\n")
    
    # Global References
    f.write(f"   [Global Top 10 Reference Entries for L{layer}-{latent}]\n")
    if ref_storage and layer < len(ref_storage) and latent < len(ref_storage[layer]):
        top_hits = ref_storage[layer][latent]
        if top_hits:
            for i, hit in enumerate(top_hits[:10]):
                f.write(f"   {i+1}. {format_global_hit(hit)}\n")
        else:
            f.write("   (No activations recorded in dataset)\n")
    else:
        f.write("   (Reference not loaded or index out of bounds)\n")
    f.write("\n")

# -----------------------------------------------------------------------------
# Website Helper Functions
# -----------------------------------------------------------------------------

def convert_to_json_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(x) for x in obj]
    return obj

def load_dataset_reference(path):
    if not path or not os.path.exists(path):
        print(f"Warning: Reference file {path} not found.")
        return None
    print(f"Loading dataset reference from {path}...")
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        return data.get("storage", [])
    except Exception as e:
        print(f"Failed to load reference file: {e}")
        return None

def write_top_activations_json(output_path, nodes_set, ref_storage, family_name):
    """Generates a top_activations.json file for a given set of nodes."""
    top_acts = {"family": family_name, "layers": {}}
    
    nodes_by_layer = {}
    for l, lat in nodes_set:
        if str(l) not in nodes_by_layer: nodes_by_layer[str(l)] = []
        nodes_by_layer[str(l)].append(lat)

    for l_str, lat_list in nodes_by_layer.items():
        top_acts["layers"][l_str] = {}
        l_idx = int(l_str)
        if ref_storage and l_idx < len(ref_storage):
            for lat in lat_list:
                hits = ref_storage[l_idx][lat] if lat < len(ref_storage[l_idx]) else []
                clean_hits = []
                for h in hits:
                    h_copy = h.copy()
                    if 'Peak_Index' in h_copy: h_copy['Peak_Index'] = max(0, h_copy['Peak_Index'] - 1)
                    if 'peak_idx' in h_copy: h_copy['peak_idx'] = max(0, h_copy['peak_idx'] - 1)
                    if 'Activations' in h_copy:
                        acts = h_copy['Activations']
                        if isinstance(acts, (list, np.ndarray, torch.Tensor)) and len(acts) > 0:
                            h_copy['Activations'] = acts[1:]
                    clean_hits.append(h_copy)
                top_acts["layers"][l_str][str(lat)] = convert_to_json_serializable(clean_hits)
        else:
             for lat in lat_list:
                 top_acts["layers"][l_str][str(lat)] = []

    with open(output_path, 'w') as f:
        json.dump(top_acts, f, indent=2)

def write_activation_indices_json(output_path, nodes_set, layer_acts_cache, start_pos, seq_len):
    """Generates activation_indices.json for a specific sequence."""
    act_ind = []
    t_start = start_pos + 1
    
    for l_idx, layer_acts in layer_acts_cache.items():
        # Filter nodes relevant to this layer
        target_latents = [lat for (l, lat) in nodes_set if l == l_idx]
        if not target_latents: continue
        
        # Iterate sequence length
        for i in range(seq_len):
            t_idx = t_start + i
            if t_idx >= layer_acts.shape[0]: break
            
            values = layer_acts[t_idx]
            for latent_id in target_latents:
                val = float(values[latent_id])
                if abs(val) > 0.001:
                    act_ind.append([l_idx, i, val, int(latent_id)])
                    
    with open(output_path, 'w') as f:
        json.dump(act_ind, f)

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Differential circuit analysis between two or three sequences.")
    parser.add_argument("--sequence1", type=str, required=True, help="Wildtype sequence")
    parser.add_argument("--sequence2", type=str, required=True, help="Positive sequence")
    parser.add_argument("--sequence3", type=str, default=None, help="Negative sequence (optional)")
    parser.add_argument("--entry_name", type=str, required=True, help="Entry Name (creates output folder)")
    parser.add_argument("--circuit_json", type=str, required=True, help="Path to circuit discovery JSON")
    parser.add_argument("--activations_pt", type=str, default="top10_activations.pt", help="Path to global activations.pt")
    
    # Cropping Arguments
    parser.add_argument("--start_pos", type=int, default=None, help="Start position (1-indexed, inclusive)")
    parser.add_argument("--end_pos", type=int, default=None, help="End position (1-indexed, inclusive)")
    
    # Model paths
    parser.add_argument("--clt_ckpt", type=str, default="../models/CLT_L6_D3200/checkpoints/last.ckpt")
    parser.add_argument("--esm_path", type=str, default="../models/esm2_t6_8M_UR50D.pt")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Prepare Output Directories
    base_dir = args.entry_name
    seq1_dir = os.path.join(base_dir, "seq1")
    seq2_dir = os.path.join(base_dir, "seq2")
    if args.sequence3 is not None:
        seq3_dir = os.path.join(base_dir, "seq3")
        os.makedirs(seq3_dir, exist_ok=True)
        sequences = [args.sequence1, args.sequence2, args.sequence3]
    else:
        sequences = [args.sequence1, args.sequence2]
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(seq1_dir, exist_ok=True)
    os.makedirs(seq2_dir, exist_ok=True)
    if args.sequence3 is not None:
        print(f"Output directories created: {seq1_dir}, {seq2_dir}, {seq3_dir}")
    else:
        print(f"Output directories created: {seq1_dir}, {seq2_dir}")

    # 2. Handle Cropping Logic (Applied to all for consistency)
    cropped_sequences = []
    
    for seq in sequences:
        s_idx = 0
        e_idx = len(seq)
        if args.start_pos is not None: s_idx = max(0, args.start_pos - 1)
        if args.end_pos is not None: e_idx = min(len(seq), args.end_pos)
        cropped_sequences.append(seq[s_idx:e_idx])

    if args.sequence3 is not None:
        seq1_cropped, seq2_cropped, seq3_cropped = cropped_sequences
    else:
        seq1_cropped, seq2_cropped = cropped_sequences
    print(f"Analysis Window: {args.start_pos if args.start_pos else 1} to {args.end_pos if args.end_pos else 'End'}")
    
    # Write seq.txt for all
    with open(os.path.join(seq1_dir, "seq.txt"), "w") as f: f.write(seq1_cropped)
    with open(os.path.join(seq2_dir, "seq.txt"), "w") as f: f.write(seq2_cropped)
    if args.sequence3 is not None:
        with open(os.path.join(seq3_dir, "seq.txt"), "w") as f: f.write(seq3_cropped)

    # 3. Load Data & Model
    print(f"Loading circuit from {args.circuit_json}...")
    with open(args.circuit_json, 'r') as f:
        circuit_data = json.load(f)
    
    circuit_nodes_map = {str(k): set(v) for k, v in circuit_data.get("nodes", {}).items()}
    ref_storage = load_dataset_reference(args.activations_pt)

    print("Loading Model...")
    pl_module = CLTLightningModule.load_from_checkpoint(args.clt_ckpt, esm2_weight=args.esm_path, strict=False)
    pl_module.to(device).eval()
    replacement_model = LocalCLTReplacementModel(pl_module, device, base_prompt=args.sequence1)

    # 4. Run Inference
    print("Running Inference on Sequence 1...")
    latents_1 = replacement_model.latents_cache # Already run by init with base_prompt
    
    print("Running Inference on Sequence 2...")
    replacement_model.update_base_prompt(args.sequence2)
    latents_2 = replacement_model.latents_cache
    
    if args.sequence3 is not None:
        print("Running Inference on Sequence 3...")
        replacement_model.update_base_prompt(args.sequence3)
        latents_3 = replacement_model.latents_cache
        diff_13_scores = []
        seq3_scores = []
        acts_cache_3 = {}

    # 5. Calculate Scores (Diff + Single Seqs)
    print("Calculating Scores...")
    
    num_layers = len(latents_1)
    
    # Lists to store scores
    diff_12_scores = []
    seq1_scores = []
    seq2_scores = []
    
    # Caches for activation generation later
    acts_cache_1 = {}
    acts_cache_2 = {}
    for layer_idx in range(num_layers):
        # Extract tensors (SeqLen, Dim)
        l1_tensor = latents_1[layer_idx].squeeze(1).cpu().numpy()
        l2_tensor = latents_2[layer_idx].squeeze(1).cpu().numpy()
        if args.sequence3 is not None:
            l3_tensor = latents_3[layer_idx].squeeze(1).cpu().numpy()
        
        acts_cache_1[layer_idx] = l1_tensor
        acts_cache_2[layer_idx] = l2_tensor
        if args.sequence3 is not None:
            acts_cache_3[layer_idx] = l3_tensor
        
        # Crop bounds
        start_crop = max(0, (args.start_pos - 1)) if args.start_pos else 0
        t_start = start_crop + 1
        
        # Circuit Latents
        circuit_latents = circuit_nodes_map.get(str(layer_idx), set())
        dim = l1_tensor.shape[1]
        
        for latent_id in range(dim):
            # Helper to process single seq
            def get_seq_stats(tensor, s_end_len):
                t_end = min(len(tensor), (args.end_pos + 1) if args.end_pos else len(tensor) + 1)
                
                if t_end > t_start and t_end <= tensor.shape[0] + 1:
                    trace = tensor[t_start:t_end, latent_id]
                    score = float(np.max(trace)) if trace.size > 0 else 0.0
                    full_trace = tensor[:, latent_id]
                    disp_trace = np.insert(full_trace[t_start:t_end], 0, 0.0)
                    peak_idx = int(np.argmax(trace)) + 1 if trace.size > 0 else 0
                    return score, disp_trace, peak_idx
                return 0.0, [], 0

            max1, disp1, peak1 = get_seq_stats(l1_tensor, len(args.sequence1))
            max2, disp2, peak2 = get_seq_stats(l2_tensor, len(args.sequence2))
            if args.sequence3 is not None:
                max3, disp3, peak3 = get_seq_stats(l3_tensor, len(args.sequence3))

            diff12 = abs(max1 - max2)
            if args.sequence3 is not None:
                diff13 = abs(max1 - max3)
            in_circuit = latent_id in circuit_latents
            
            # Populate Lists
            # Diff 1-2
            diff_12_scores.append({
                'layer': layer_idx, 'latent': latent_id, 'diff': diff12, 'in_circuit': in_circuit,
                'seqA_data': {'score': max1, 'trace': disp1, 'peak_idx': peak1},
                'seqB_data': {'score': max2, 'trace': disp2, 'peak_idx': peak2}
            })
            # Diff 1-3
            if args.sequence3 is not None:
                diff_13_scores.append({
                    'layer': layer_idx, 'latent': latent_id, 'diff': diff13, 'in_circuit': in_circuit,
                    'seqA_data': {'score': max1, 'trace': disp1, 'peak_idx': peak1},
                    'seqB_data': {'score': max3, 'trace': disp3, 'peak_idx': peak3}
                })
            # Singles
            if max1 > 0:
                seq1_scores.append({'layer': layer_idx, 'latent': latent_id, 'score': max1, 'in_circuit': in_circuit, 'trace': disp1, 'peak_idx': peak1})
            if max2 > 0:
                seq2_scores.append({'layer': layer_idx, 'latent': latent_id, 'score': max2, 'in_circuit': in_circuit, 'trace': disp2, 'peak_idx': peak2})
            if args.sequence3 is not None and max3 > 0:
                seq3_scores.append({'layer': layer_idx, 'latent': latent_id, 'score': max3, 'in_circuit': in_circuit, 'trace': disp3, 'peak_idx': peak3})

    # --- 6. Select Top Nodes ---
    def select_top_nodes_generic(score_list, score_key, target_per_layer, top_k_global=20):
        selected = []
        seen = set()
        score_list.sort(key=lambda x: x[score_key], reverse=True)
        
        # Top Global
        for item in score_list[:top_k_global]:
            key = (item['layer'], item['latent'])
            if key not in seen:
                seen.add(key)
                selected.append(item)
        
        # Backfill per layer
        for layer in range(num_layers):
            layer_items = [x for x in score_list if x['layer'] == layer]
            count = sum(1 for x in selected if x['layer'] == layer)
            if count < target_per_layer:
                needed = target_per_layer - count
                added = 0
                for item in layer_items:
                    key = (item['layer'], item['latent'])
                    if key not in seen:
                        seen.add(key)
                        selected.append(item)
                        added += 1
                        if added >= needed: break
        
        selected.sort(key=lambda x: x[score_key], reverse=True)
        return selected

    print("Selecting top nodes...")
    final_diff12_nodes = select_top_nodes_generic(diff_12_scores, 'diff', TARGET_PER_LAYER_DIFF, top_k_global=20)
    if args.sequence3 is not None:
        final_diff13_nodes = select_top_nodes_generic(diff_13_scores, 'diff', TARGET_PER_LAYER_DIFF, top_k_global=20)
        final_seq3_nodes = select_top_nodes_generic(seq3_scores, 'score', TARGET_PER_LAYER_SINGLE, top_k_global=10)
    
    final_seq1_nodes = select_top_nodes_generic(seq1_scores, 'score', TARGET_PER_LAYER_SINGLE, top_k_global=10)
    final_seq2_nodes = select_top_nodes_generic(seq2_scores, 'score', TARGET_PER_LAYER_SINGLE, top_k_global=10)
    # 7. Generate Unified JSONs for Website
    print("Generating Unified JSONs (Superset of all analyses)...")
    
    # Union of ALL interesting nodes found in ANY step
    global_union_nodes = set()
    for item in final_diff12_nodes: global_union_nodes.add((item['layer'], item['latent']))
    if args.sequence3 is not None:
        for item in final_diff13_nodes: global_union_nodes.add((item['layer'], item['latent']))
        for item in final_seq3_nodes: global_union_nodes.add((item['layer'], item['latent']))
    for item in final_seq1_nodes: global_union_nodes.add((item['layer'], item['latent']))
    for item in final_seq2_nodes: global_union_nodes.add((item['layer'], item['latent']))

    fam_name = circuit_data.get("family", "Unknown")
    s_crop_start = max(0, (args.start_pos - 1)) if args.start_pos else 0

    # Write for Seq 1
    write_top_activations_json(os.path.join(seq1_dir, "top_activations.json"), global_union_nodes, ref_storage, fam_name)
    write_activation_indices_json(os.path.join(seq1_dir, "activation_indices.json"), global_union_nodes, acts_cache_1, s_crop_start, len(seq1_cropped))

    # Write for Seq 2
    write_top_activations_json(os.path.join(seq2_dir, "top_activations.json"), global_union_nodes, ref_storage, fam_name)
    write_activation_indices_json(os.path.join(seq2_dir, "activation_indices.json"), global_union_nodes, acts_cache_2, s_crop_start, len(seq2_cropped))

    if args.sequence3 is not None:
        # Write for Seq 3
        write_top_activations_json(os.path.join(seq3_dir, "top_activations.json"), global_union_nodes, ref_storage, fam_name)
        write_activation_indices_json(os.path.join(seq3_dir, "activation_indices.json"), global_union_nodes, acts_cache_3, s_crop_start, len(seq3_cropped))

    # 8. Comparison Reports
    def write_diff_report(path, nodes_list, title, sA, sB):
        with open(path, "w") as f:
            f.write(f"=== {title} ===\n")
            f.write(f"Total Diff Nodes Analyzed: {len(nodes_list)}\n")
            f.write("-------------------------------------\n\n")
            for i, item in enumerate(nodes_list):
                rank_label = f"Rank #{i+1}"
                if item['in_circuit']: rank_label += " (In Circuit JSON)"
                else: rank_label += " (Backfill)"
                print_differential_result_block(f, rank_label, item, sA, sB, ref_storage)

    print("Generating Differential Reports...")
    write_diff_report(os.path.join(base_dir, "analysis_differential_12.txt"), final_diff12_nodes, "Differential Analysis: WT vs Positive", seq1_cropped, seq2_cropped)
    if args.sequence3 is not None:
        write_diff_report(os.path.join(base_dir, "analysis_differential_13.txt"), final_diff13_nodes, "Differential Analysis: WT vs Negative", seq1_cropped, seq3_cropped)

    # 9. Single Sequence Reports
    def write_single_report(path, nodes_list, title, seq):
        with open(path, "w") as f:
            f.write(f"=== {title} ===\n")
            f.write(f"Total Nodes: {len(nodes_list)}\n")
            f.write("-------------------------------------\n\n")
            for i, item in enumerate(nodes_list):
                rank_label = f"Rank #{i+1}"
                if item['in_circuit']: rank_label += " (In Circuit JSON)"
                else: rank_label += " (Backfill)"
                print_result_block(f, rank_label, item, seq, ref_storage)

    print("Generating Single Sequence Reports...")
    write_single_report(os.path.join(base_dir, "top_seq1.txt"), final_seq1_nodes, "Top Activations: WT (Seq1)", seq1_cropped)
    write_single_report(os.path.join(base_dir, "top_seq2.txt"), final_seq2_nodes, "Top Activations: Positive (Seq2)", seq2_cropped)
    if args.sequence3 is not None:
        write_single_report(os.path.join(base_dir, "top_seq3.txt"), final_seq3_nodes, "Top Activations: Negative (Seq3)", seq3_cropped)

    print("Done!")

if __name__ == "__main__":
    main()