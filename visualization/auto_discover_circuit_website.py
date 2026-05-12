import sys
import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

script_dir = os.path.dirname(os.path.abspath(__file__)) 
repo_root = os.path.dirname(script_dir)                 
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "training"))
sys.path.append(os.path.join(repo_root, "family_circuit"))
sys.path.append(os.path.join(repo_root, "function_circuit"))

try:
    from circuit_utils.esm_activation import ESMInference
    from circuit_utils.clt_circuit import CircuitDiscovererCLT
    from circuit_utils.circuit_utils import compute_attribution, rank_nodes, circuit_search
    
    # Binary Utils
    from family_utils import train_probe as train_linear_probe
    from family_utils import evaluate_circuit as evaluate_circuit_binary 
    
    # Regression Utils
    from function_utils import (
        CNNProbe, train_probe_cnn, evaluate_probe_direct, 
        EmbeddingDataset, evaluate_circuit as evaluate_circuit_regression,
        precompute_embeddings
    )

except ImportError as e:
    print(f"❌ Error importing project modules: {e}")
    sys.exit(1)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class MemmapSubset:
    """Helper to slice a master memmap without copying data."""
    def __init__(self, memmap_obj, indices):
        self.memmap = memmap_obj
        self.indices = indices
    
    def __getitem__(self, idx):
        global_indices = self.indices[idx] 
        return self.memmap[global_indices]
    
    def __len__(self):
        return len(self.indices)

def find_column(df, keyword):
    cols = df.columns
    for c in cols:
        if keyword.lower() in c.lower():
            return c
    return None

def load_and_parse_csv(csv_path, is_binary):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    seq_col = find_column(df, "sequence")
    if not seq_col:
        if "mutated_sequence" in df.columns: seq_col = "mutated_sequence"
        else: raise ValueError(f"Could not find a column containing 'sequence' in {df.columns}")
    
    if is_binary:
        score_col = find_column(df, "bin") 
        if not score_col: score_col = find_column(df, "class")
        if not score_col: score_col = find_column(df, "score")
    else:
        score_col = find_column(df, "score")
        if not score_col: score_col = find_column(df, "target")

    if not score_col:
        raise ValueError(f"Could not find a score/target column in {df.columns}")
    
    seqs = df[seq_col].values.astype(str)
    scores = df[score_col].values

    if is_binary:
        unique_vals = np.unique(scores)
        if not np.all(np.isin(unique_vals, [0, 1])):
            if scores.dtype == bool:
                scores = scores.astype(int)
            else:
                raise ValueError(f"Binary mode enabled, but column '{score_col}' has non-binary values.")
        scores = scores.astype(int)
    else:
        if not np.issubdtype(scores.dtype, np.number):
             raise ValueError(f"Regression mode enabled, but column '{score_col}' is not numeric.")
        scores = scores.astype(float)

    return seqs, scores

def main():
    parser = argparse.ArgumentParser(description="Auto-Discover Circuits")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--is_binary", type=lambda x: (str(x).lower() == 'true'), required=True)
    parser.add_argument("--output_dir", type=str, default="auto_circuits")
    parser.add_argument("--entry_name", type=str, default="experiment")
    parser.add_argument("--model_size", type=str, choices=["ESM2-8M", "ESM2-35M"], default="ESM2-35M")
    parser.add_argument("--clt_checkpoint", type=str, default=None)
    parser.add_argument("--esm_weights", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_nodes", type=int, default=1000)
    parser.add_argument("--step_size", type=int, default=32)
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 0. Validate model config and defaults
    if args.model_size == "ESM2-8M":
        default_esm = Path(repo_root, "models", "esm2_t6_8M_UR50D.pt")
        default_clt = Path(repo_root, "models", "CLT_L6_D3200", "checkpoints", "last.ckpt")
        expected_n_layers = 6
        expected_d_model = 320
    else:
        default_esm = Path(repo_root, "models", "esm2_t12_35M_UR50D.pt")
        default_clt = Path(repo_root, "models", "CLT_L12_D4800", "checkpoints", "last.ckpt")
        expected_n_layers = 12
        expected_d_model = 480

    if args.esm_weights is None:
        args.esm_weights = default_esm
    if args.clt_checkpoint is None:
        args.clt_checkpoint = default_clt

    esm_name = Path(args.esm_weights).name
    if args.model_size == "ESM2-8M" and "35M" in esm_name:
        raise ValueError(f"Model size mismatch: model_size={args.model_size} but esm_weights={args.esm_weights}")
    if args.model_size == "ESM2-35M" and "8M" in esm_name:
        raise ValueError(f"Model size mismatch: model_size={args.model_size} but esm_weights={args.esm_weights}")

    # 1. Load Data
    all_seqs, all_scores = load_and_parse_csv(args.csv_path, args.is_binary)
    
    # 2. Init Models
    weight_path = Path(args.esm_weights).name
    layer_match = re.search(r't(\d+)', weight_path)
    n_layers = int(layer_match.group(1)) if layer_match else expected_n_layers
    param_match = re.search(r'_(\d+M)_', weight_path)
    param_count = param_match.group(1) if param_match else ""
    if param_count == "35M":
        d_model = 480
    elif param_count == "8M":
        d_model = 320
    else:
        d_model = expected_d_model
    inference = ESMInference(device, esm_weights_path=args.esm_weights, num_layers=n_layers, d_model=d_model)
    discoverer = CircuitDiscovererCLT(device, ckpt_path=args.clt_checkpoint, esm_weights_path=args.esm_weights)

    # 3. Pipeline Logic
    source_layer = "mlp_output"
    freeze_attention = True
    
    best_nodes = {}
    best_k = 0
    max_metrics = {}
    base_metrics = {}
    final_metrics = {}
    recovered_ratio = 0.0
    metric_name = ""
    clean_metric_test = 0.0
    recovered_val = 0.0
    n_train_samples = 0

    if args.is_binary:
        print("=== Mode: Binary Classification ===")
        metric_name = "F1"            
        seq_train_full, seq_test, y_train_full, y_test = train_test_split(all_seqs, all_scores, test_size=0.2, random_state=42)
        seq_train, seq_val, y_train, y_val = train_test_split(seq_train_full, y_train_full, test_size=0.1, random_state=42)
        n_train_samples = len(seq_train)

        # Extract embeddings
        def get_embs(sequences):
            return inference.get_embeddings(sequences.tolist(), source=source_layer, mean_pool=True)
        X_train = get_embs(seq_train)
        X_val = get_embs(seq_val)
        X_test = get_embs(seq_test)

        # Train linear probe.
        probe_out = train_linear_probe(X_train, y_train, device)
        scaler = None
        clf = None
        if isinstance(probe_out, tuple):
            probe = probe_out[0]
            if len(probe_out) >= 3:
                scaler = probe_out[1]
                clf = probe_out[2]
        else:
            probe = probe_out

        probe_save_path = os.path.join(args.output_dir, f"{args.entry_name}_probe.pt")
        torch.save(probe.state_dict(), probe_save_path)
        print(f"Saved binary probe to {probe_save_path}")

        def predict_binary_from_embeddings(X):
            if clf is not None and scaler is not None:
                return clf.predict(scaler.transform(X))
            with torch.no_grad():
                x_t = torch.as_tensor(X, device=device, dtype=torch.float)
                logits = probe(x_t)
                if logits.dim() > 1 and logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                if logits.dim() > 1:
                    logits = logits.mean(dim=1)
                return (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int)

        y_val_pred = predict_binary_from_embeddings(X_val)
        clean_metric = f1_score(y_val, y_val_pred)

        # Compute attribution
        pos_val_mask = (y_val == 1)
        pos_seqs_for_attr = seq_val if pos_val_mask.sum() == 0 else seq_val[pos_val_mask]
        global_attr = compute_attribution(
            discoverer, probe, pos_seqs_for_attr, batch_size=args.batch_size, 
            sequential=True, freeze_attention=freeze_attention, source=source_layer
        )
        ranking = rank_nodes(global_attr)

        # Disover circuit
        def eval_fn_binary(d, p, s, y_true, nodes, bs):
            return evaluate_circuit_binary(d, p, s, y_true, nodes, bs, sequential=True, freeze_attention=freeze_attention, source=source_layer)['f1']
        best_nodes, best_k, _ = circuit_search(
            discoverer, probe, ranking, seq_val, y_val,
            target_metric=clean_metric * 0.7,
            metric_fn=eval_fn_binary,
            step_size=args.step_size,
            max_nodes=args.max_nodes,
            desc="Binary Search"
        )

        # Evaluate test set
        clean_metric_test = f1_score(y_test, predict_binary_from_embeddings(X_test))
        max_metrics = evaluate_circuit_binary(discoverer, probe, seq_test, y_test, None, args.batch_size, gt_embeddings=X_test, sequential=True, freeze_attention=freeze_attention, source=source_layer)
        base_metrics = evaluate_circuit_binary(discoverer, probe, seq_test, y_test, {}, args.batch_size, gt_embeddings=X_test, sequential=True, freeze_attention=freeze_attention, source=source_layer)
        final_metrics = evaluate_circuit_binary(discoverer, probe, seq_test, y_test, best_nodes, args.batch_size, gt_embeddings=X_test, sequential=True, freeze_attention=freeze_attention, source=source_layer)
        recovered_val = final_metrics['f1']

    else:
        print("=== Mode: Regression ===")
        metric_name = "Spearman"
        
        # Save to disk to preserve memory
        cache_root = Path(f"embeddings_cache_{args.entry_name}").resolve()
        os.makedirs(cache_root, exist_ok=True)
        dms_name = args.entry_name
        target_layer = n_layers - 1 
        cache_filename = f"{dms_name}_all_L{target_layer}_mlp_seq.dat"
        cache_path = cache_root / cache_filename
        if cache_path.exists():
            print(f"Found existing embeddings cache at {cache_path}. Loading...")
            cache_dir = cache_path
        else:
            print(f"Computing embeddings for {len(all_seqs)} sequences...")
        print(cache_root)
        master_embeddings = precompute_embeddings(
            all_seqs, 
            inference, 
            dms_name, 
            target_layer=target_layer, 
            cache_dir=str(cache_root), 
            batch_size=args.batch_size
        )
        
        # Split Indices
        all_indices = np.arange(len(all_seqs))
        train_idx_full, test_idx, y_train_full, y_test = train_test_split(all_indices, all_scores, test_size=0.2, random_state=42)
        train_idx, val_idx, y_train, y_val = train_test_split(train_idx_full, y_train_full, test_size=0.1, random_state=42)
        n_train_samples = len(train_idx)

        # Datasets for Probe Training
        train_ds = EmbeddingDataset(master_embeddings, y_train, indices=train_idx)
        val_ds = EmbeddingDataset(master_embeddings, y_val, indices=val_idx)
        test_ds = EmbeddingDataset(master_embeddings, y_test, indices=test_idx)
        
        # Subset for Circuit Evaluation
        test_subset_embs = MemmapSubset(master_embeddings, test_idx)
        seq_val = all_seqs[val_idx]
        seq_test = all_seqs[test_idx]

        # Train CNN probe
        input_dim = inference.model.embed_dim
        cnn_probe = CNNProbe(input_dim).to(device)
        cnn_probe, _, _, _ = train_probe_cnn(
            train_ds, val_ds, input_dim, device,
            total_steps=10000, 
            batch_size=args.batch_size,
            recycle_data=True,
            patience=10
        )
        probe_save_path = os.path.join(args.output_dir, f"{args.entry_name}_probe.pt")
        torch.save(cnn_probe.state_dict(), probe_save_path)
        print(f"Saved regression probe to {probe_save_path}")
        clean_metric = evaluate_probe_direct(cnn_probe, val_ds, y_val, device, batch_size=args.batch_size)

        # Compute attribution
        median_score = np.median(y_val)
        top_indices = np.where(y_val >= median_score)[0]
        seq_attr = seq_val[top_indices] if len(top_indices) > 0 else seq_val
        global_attr = compute_attribution(
            discoverer, cnn_probe, seq_attr, 
            batch_size=args.batch_size, 
            sequential=True, freeze_attention=freeze_attention, source=source_layer
        )
        ranking = rank_nodes(global_attr)

        # Discover circuit
        def eval_fn_reg(d, p, s, y_true, nodes, bs):
            return evaluate_circuit_regression(d, p, s, y_true, nodes, bs, cnn=True, sequential=True, freeze_attention=freeze_attention, source=source_layer)['spearman']
        best_nodes, best_k, _ = circuit_search(
            discoverer, cnn_probe, ranking, seq_val, y_val,
            target_metric=clean_metric * 0.7,
            metric_fn=eval_fn_reg,
            step_size=args.step_size,
            max_nodes=args.max_nodes,
            desc="Regression Search"
        )

        # Evaluate test set
        clean_metric_test = evaluate_probe_direct(cnn_probe, test_ds, y_test, device, batch_size=args.batch_size)
        max_metrics = evaluate_circuit_regression(discoverer, cnn_probe, seq_test, y_test, None, args.batch_size, cnn=True, gt_embeddings=test_subset_embs, sequential=True, freeze_attention=freeze_attention, source=source_layer)
        base_metrics = evaluate_circuit_regression(discoverer, cnn_probe, seq_test, y_test, {}, args.batch_size, cnn=True, gt_embeddings=test_subset_embs, sequential=True, freeze_attention=freeze_attention, source=source_layer)
        final_metrics = evaluate_circuit_regression(discoverer, cnn_probe, seq_test, y_test, best_nodes, args.batch_size, cnn=True, gt_embeddings=test_subset_embs, sequential=True, freeze_attention=freeze_attention, source=source_layer)
        recovered_val = final_metrics['spearman']

    # --- Save Results ---
    denom = (clean_metric_test if args.is_binary else max_metrics[metric_name.lower()]) - base_metrics[metric_name.lower()]
    if denom > 1e-6:
        recovered_ratio = (recovered_val - base_metrics[metric_name.lower()]) / denom

    print(f"\nResult ({metric_name}):")
    print(f"Clean/Max: {clean_metric_test:.4f}")
    print(f"Circuit:   {recovered_val:.4f}")
    print(f"Ratio:     {recovered_ratio:.4f}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.entry_name}.json")
    
    result_data = {
        "family": args.entry_name,
        "n_train": n_train_samples,
        "k": int(best_k),
        "source": source_layer,
        "freeze_attention": freeze_attention,
        "clean_metric": float(clean_metric_test),
        "max_metric": float(max_metrics.get(metric_name.lower(), 0.0)),
        "recovered_metric": float(recovered_val),
        "base_metric": float(base_metrics.get(metric_name.lower(), 0.0)),
        "recovered_ratio": float(recovered_ratio),
        "nodes": {str(l): list(s) for l, s in best_nodes.items()}
    }
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)


if __name__ == "__main__":
    main()