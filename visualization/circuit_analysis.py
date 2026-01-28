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

# Import ESM2ActivationCollector
try:
    from esm_activation import ESM2ActivationCollector
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_utils"))
        from esm_activation import ESM2ActivationCollector
    except ImportError as e:
        print(f"Warning: ESM2ActivationCollector import failed: {e}")
        ESM2ActivationCollector = None

# Import CLT Module
try:
    from clt_module import CLTLightningModule
except ImportError as e:
    # Fallback/Mock for environment check, in prod ensure this path is correct
    print(f"Warning: CLTLightningModule import failed: {e}")
    CLTLightningModule = None

#from full_replacement_models import FullCLTReplacementModel
from local_replacement_models import LocalCLTReplacementModel

# -----------------------------------------------------------------------------
# Analysis Helper Functions (From analyze_sequence_circuit_dev.py)
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
    layer = item['layer']
    latent = item['latent']
    score = item['score']
    trace = item['trace']
    peak_idx = item['peak_idx']
    
    # Adaptive Window
    motif_str, seq_start, seq_end = get_adaptive_motif_window(
        sequence, trace, peak_idx, min_radius=5, buffer=3, top_k=None 
    )
    
    # Trace Window
    t_start = max(0, seq_start + 1)
    t_end = min(len(trace), seq_end + 1)
    trace_window = trace[t_start:t_end]
    
    f.write(f"--- Rank {rank_label} ---\n")
    f.write(f"Node: Layer {layer}, Latent {latent}\n")
    f.write(f"Max Activation: {score:.4f}\n")
    f.write(f"Peak Location: Trace Idx {peak_idx} (AA #{peak_idx})\n")
    f.write(f"Motif Context: {motif_str}\n")
    f.write(f"Trace Window : {trace_window}\n")
    f.write("\n")
    
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

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive circuit analysis assets.")
    parser.add_argument("--sequence", type=str, required=True, help="The protein sequence")
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

    # 1. Prepare Output Directory
    output_dir = args.entry_name
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 2. Handle Cropping Logic
    full_sequence = args.sequence
    full_len = len(full_sequence)
    
    # Convert 1-indexed args to 0-indexed slice
    s_idx = 0
    e_idx = full_len
    
    if args.start_pos is not None:
        s_idx = max(0, args.start_pos - 1)
    if args.end_pos is not None:
        e_idx = min(full_len, args.end_pos)
        
    cropped_sequence = full_sequence[s_idx:e_idx]
    print(f"Full Seq Len: {full_len}. Analysis Window: {s_idx+1} to {e_idx} (Len: {len(cropped_sequence)})")

    # 3. Write seq.txt (Cropped)
    seq_path = os.path.join(output_dir, "seq.txt")
    with open(seq_path, "w") as f:
        f.write(cropped_sequence)
    print(f"Generated {seq_path}")

    # 4. Load Data & Model
    print(f"Loading circuit from {args.circuit_json}...")
    with open(args.circuit_json, 'r') as f:
        circuit_data = json.load(f)
    
    circuit_nodes = {str(k): v for k, v in circuit_data.get("nodes", {}).items()}
    circuit_layers_in_json = set(int(k) for k in circuit_nodes.keys())
    ref_storage = load_dataset_reference(args.activations_pt)

    print("Loading Model...")
    pl_module = CLTLightningModule.load_from_checkpoint(args.clt_ckpt, esm2_weight=args.esm_path, strict=False)
    pl_module.to(device).eval()
    replacement_model = LocalCLTReplacementModel(pl_module, device, base_prompt=full_sequence)
    # replacement_model = FullCLTReplacementModel(pl_module, device)

    # 5. Run Inference (On FULL Sequence to preserve context)
    print("Running Inference on full sequence...")
    with torch.no_grad():
        _, latents_list_L, _, _, _ = replacement_model.forward([full_sequence], freeze_attention=True)

    # 6. Score and Select Nodes (Based on CROPPED Window)
    print("Calculating scores within window...")
    all_node_scores = []
    layer_acts_cache = {} 

    for layer_str, latent_ids in circuit_nodes.items():
        layer_idx = int(layer_str)
        if layer_idx >= len(latents_list_L): continue
            
        # Get activations: (SeqLen, Dim)
        layer_tensor = latents_list_L[layer_idx]
        layer_acts = layer_tensor.squeeze(1).cpu().numpy()
        layer_acts_cache[layer_idx] = layer_acts

        # Score based on VALID SLICE (s_idx to e_idx)
        # Trace index = Seq index + 1 (Skip CLS)
        # Sequence indices s_idx...e_idx-1 correspond to Trace indices s_idx+1...e_idx
        trace_start = s_idx + 1
        trace_end = e_idx + 1
        
        # Clamp to bounds
        if trace_start >= layer_acts.shape[0]: continue
        trace_end = min(trace_end, layer_acts.shape[0])
        
        for latent_id in latent_ids:
            # Extract only the window we care about
            window_trace = layer_acts[trace_start:trace_end, latent_id]
            if window_trace.size > 0:
                max_val = float(np.max(window_trace))
                all_node_scores.append({
                    'layer': layer_idx, 
                    'latent': latent_id, 
                    'score': max_val,
                    # We need the relative peak index for analysis text
                    'peak_offset': int(np.argmax(window_trace)) 
                })

    # Sort descending by score
    all_node_scores.sort(key=lambda x: x['score'], reverse=True)

    # --- Selection Logic (Top 10 + 5 Per Layer) ---
    final_selected_nodes = set() 
    analysis_results_top10 = []
    analysis_results_additional = []

    # A. Top 10 Global
    for i, item in enumerate(all_node_scores[:10]):
        key = (item['layer'], item['latent'])
        final_selected_nodes.add(key)
        
        # Prepare data for analysis text
        # Peak Index logic: The argmax was relative to window_trace.
        # Global trace index = s_idx + 1 + peak_offset
        # But for 'cropped sequence' visualization, we want index relative to cropped seq?
        # Actually, get_adaptive_motif_window takes the sequence. If we pass cropped_seq, 
        # the peak_idx should be relative to that trace (1-based).
        # Let's say we pass cropped_seq. 
        # Trace for cropped seq is effectively window_trace padded with CLS.
        # So peak_idx = peak_offset + 1.
        
        # Re-extract trace for just the window
        l_idx, lat_idx = item['layer'], item['latent']
        full_trace = layer_acts_cache[l_idx][:, lat_idx]
        cropped_trace = full_trace[trace_start:trace_end]
        # Prepend a dummy 0 for CLS alignment in helper function
        display_trace = np.insert(cropped_trace, 0, 0.0) 
        
        analysis_results_top10.append({
            "layer": l_idx, "latent": lat_idx, "score": item['score'],
            "trace": display_trace,
            "peak_idx": item['peak_offset'] + 1
        })

    # B. Ensure 5 per layer
    sorted_circuit_layers = sorted(list(circuit_layers_in_json))
    for layer in sorted_circuit_layers:
        layer_candidates = [x for x in all_node_scores if x['layer'] == layer]
        current_count = sum(1 for x in layer_candidates if (x['layer'], x['latent']) in final_selected_nodes)
        
        needed = 5 - current_count
        if needed > 0:
            added = 0
            for item in layer_candidates:
                key = (item['layer'], item['latent'])
                if key not in final_selected_nodes:
                    final_selected_nodes.add(key)
                    
                    # Prepare for analysis text
                    rank_str = f" (Rank {current_count + added + 1} of Layer {layer})"
                    
                    l_idx, lat_idx = item['layer'], item['latent']
                    full_trace = layer_acts_cache[l_idx][:, lat_idx]
                    cropped_trace = full_trace[trace_start:trace_end]
                    display_trace = np.insert(cropped_trace, 0, 0.0)
                    
                    analysis_results_additional.append((
                        rank_str, 
                        {
                            "layer": l_idx, "latent": lat_idx, "score": item['score'],
                            "trace": display_trace,
                            "peak_idx": item['peak_offset'] + 1
                        }
                    ))
                    
                    added += 1
                    if added >= needed: break

    # 7. Generate top_activations.json
    print("Generating top_activations.json...")
    top_acts_out = {"family": circuit_data.get("family", "Unknown"), "layers": {}}
    
    nodes_by_layer = {}
    for l, lat in final_selected_nodes:
        if str(l) not in nodes_by_layer: nodes_by_layer[str(l)] = []
        nodes_by_layer[str(l)].append(lat)

    if ref_storage:
        for layer_str, latent_ids in nodes_by_layer.items():
            layer_idx = int(layer_str)
            top_acts_out["layers"][layer_str] = {}
            
            if layer_idx < len(ref_storage):
                for latent_id in latent_ids:
                    if latent_id < len(ref_storage[layer_idx]):
                        hits = ref_storage[layer_idx][latent_id]
                        
                        # --- FIX START: Shift Indices and Slice Trace ---
                        processed_hits = []
                        for h in hits:
                            # Copy the hit dictionary to avoid modifying the original cache
                            h_clean = h.copy()
                            
                            # 1. Shift Peak Index (1-based -> 0-based)
                            # Check both naming conventions just in case
                            if 'Peak_Index' in h_clean and h_clean['Peak_Index'] is not None:
                                h_clean['Peak_Index'] = max(0, int(h_clean['Peak_Index']) - 1)
                            if 'peak_idx' in h_clean and h_clean['peak_idx'] is not None:
                                h_clean['peak_idx'] = max(0, int(h_clean['peak_idx']) - 1)
                            
                            # 2. Slice the Trace (Remove [CLS] token at index 0)
                            if 'Activations' in h_clean:
                                acts = h_clean['Activations']
                                # Handle Tensor/Numpy conversion
                                if isinstance(acts, torch.Tensor):
                                    acts = acts.detach().cpu().tolist()
                                elif isinstance(acts, np.ndarray):
                                    acts = acts.tolist()
                                
                                # Slice off the first element (CLS)
                                if len(acts) > 1:
                                    h_clean['Activations'] = acts[1:]
                                else:
                                    h_clean['Activations'] = []
                                    
                            processed_hits.append(h_clean)
                        # --- FIX END ---
                        
                        top_acts_out["layers"][layer_str][str(latent_id)] = convert_to_json_serializable(processed_hits)
                    else:
                        top_acts_out["layers"][layer_str][str(latent_id)] = []
    
    with open(os.path.join(output_dir, "top_activations.json"), 'w') as f:
        json.dump(top_acts_out, f, indent=2)

    # 8. Generate activation_indices.json (Relative to Window)
    print("Generating activation_indices.json...")
    activation_indices = []
    
    for layer_idx, layer_acts in layer_acts_cache.items():
        target_latents = [lat for (l, lat) in final_selected_nodes if l == layer_idx]
        if not target_latents: continue

        # Loop over the CROPPED sequence indices (0 to len(cropped)-1)
        for rel_seq_pos in range(len(cropped_sequence)):
            # Map back to full trace index
            full_trace_idx = s_idx + rel_seq_pos + 1
            
            if full_trace_idx >= layer_acts.shape[0]: break
            
            values = layer_acts[full_trace_idx]
            for latent_id in target_latents:
                val = float(values[latent_id])
                if abs(val) > 0.001:
                    # Store RELATIVE sequence position
                    activation_indices.append([layer_idx, rel_seq_pos, val, int(latent_id)])

    with open(os.path.join(output_dir, "activation_indices.json"), 'w') as f:
        json.dump(activation_indices, f)

    # 9. Generate Analysis Text Report
    txt_output_path = os.path.join(output_dir, "analysis.txt")
    print(f"Generating analysis report at {txt_output_path}...")
    
    with open(txt_output_path, "w") as f:
        f.write("=== CLT Sequential Frozen Pass Analysis ===\n")
        f.write(f"Circuit Family: {circuit_data.get('family', 'Unknown')}\n")
        f.write(f"Input Entry: {args.entry_name}\n")
        f.write(f"Window Analyzed: Residues {s_idx+1} to {e_idx}\n")
        f.write(f"Sequence Length (Cropped): {len(cropped_sequence)}\n")
        f.write(f"Total Circuit Nodes Selected: {len(final_selected_nodes)}\n")
        f.write("-------------------------------------------\n\n")

        for i, item in enumerate(analysis_results_top10):
            print_result_block(f, f"#{i+1}", item, cropped_sequence, ref_storage)
            
        if analysis_results_additional:
            f.write("--- Additional Layers (Coverage up to 5 per layer) ---\n\n")
            for note, item in analysis_results_additional:
                print_result_block(f, f"ADDITIONAL{note}", item, cropped_sequence, ref_storage)

    print("Done!")

if __name__ == "__main__":
    main()