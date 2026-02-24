# Visualization Documentation

This folder contains tools for **analyzing and visualizing CLT (Cross-Layer Transcoder) circuit activations** on protein sequences. It supports both:
- **Family circuits** — Analyzing which latents respond to protein domain signature.
- **Function circuits** — Analyzing how latents encode functional properties and comparing activation patterns between wild-type and mutant sequences.

---

## Creating a Graph from a Family Circuit

1. Download the families.tar.gz file from our [HuggingFace](https://huggingface.co/datasets/ktalreja/ProtoMechData).
2. Find the sequence you want to examine and its InterPro family (IPRXXXXXX). If you don't know what
InterPro family your sequence is in, try the [InterPro Search](https://www.ebi.ac.uk/interpro/search/sequence/). Our data contains circuits for all the families in SwissProt. If you keep the `CIRCUIT_JSON` blank, it will use the top 10 activations at each layer for the given sequence.
3. Find `create_family_graph.sh`. Choose the given family circuit json under `base/families/[MODEL_TYPE]/IPRXXXXXX.json`. We recommend using CLT_sequential for the model type, but all of the ones we tested are there. Substitute your sequence in the `SEQUENCE` variable and give an output folder name in `NAME`. Run the script, preferrably with a GPU so the edge weights get done quicker.
4. You will find 4 files inside the folder called `NAME`: `activation_indices.json`, `seq.txt`, `top_activations.json`, `virtual_weights.json` (another file called `analysis.txt` which contains information about top activations will also be found, but is not necessary for creating the graph). Download them. Navigate to the website, hit "Load Custom Circuit" at the top, and input these files.

---

## Creating a Graph from a Function Circuit

1. Download the functions.tar.gz file from our [HuggingFace](https://huggingface.co/datasets/ktalreja/ProtoMechData).
2. Find `create_function_graph.sh`. Substitute your sequences in the `SEQUENCE1`, `SEQUENCE2`, etc. variables and give an output folder name in `NAME`. Choose the function circuit you want to use (note that the pre-loaded circuits are for the 12 DMS assays we ran) in `CIRCUIT_JSON`, or else keep it blank and the circuit will default to the top 10 activations for `SEQUENCE1` at each layer. Enter the sequences you want for comparison, and run the script. You can use 2 or 3 sequences to do this. Run the script, preferrably with a GPU so the edge weights get done quicker.
3. This will create a set of folders under `NAME` that contain the `activation_indices.json`, `seq.txt`, `top_activations.json`, `virtual_weights.json` for each sequence in `NAME/seq1`, `NAME/seq2`, etc. It will also contain files of the form `top_seq1.txt`, `top_seq2.txt`, etc. which contains the top-activating latents for the given sequences, along with other related proteins. It will also contain `analysis_differential{i}{j}.txt` that shows which nodes differ the most between sequences.

---

## Core Analysis Scripts

### `circuit_analysis.py`
**Purpose:** Analyzes CLT circuit activations on a single protein sequence using a family circuit.

**Key Features:**
- Runs the sequence through the CLT model
- Extracts activations for all latents specified in the circuit JSON
- Identifies peak activation positions in the sequence
- Generates a detailed text report with motif windows

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--sequence` | The protein sequence to analyze |
| `--circuit_json` | Path to the circuit JSON (list of `[layer, latent_idx]` pairs) |
| `--entry_name` | Name for the output directory and files |
| `--output` | (Optional) Custom output filename |

**Output Files:**
- `analysis.txt` — Detailed report of top activating latents per layer
- `activation_indices.json` — Machine-readable activation data for downstream tools
- `top_activations.json` — Global top activations for website/visualization integration
- `seq.txt` — The input sequence

---

### `circuit_analysis_function.py`
**Purpose:** Compares circuit activations between 3 sequences to identify **differential activations** — latents that respond differently to mutant vs. wild-type sequences.

**Key Features:**
- Supports comparing up to 3 sequences (e.g., WT, positive mutant, negative mutant)
- Computes per-node activation differences
- Generates both single-sequence reports and differential analysis reports
- Outputs JSON for website integration

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--sequence1` | First sequence (typically wild-type) |
| `--sequence2` | Second sequence for comparison |
| `--sequence3` | Third sequence for 3-way comparison (optional) |
| `--circuit_json` | Path to the circuit JSON |
| `--entry_name` | Name for the output directory |
| `--start_pos` / `--end_pos` | (Optional) Restrict analysis to a subsequence |

**Output Structure:**
```
<entry_name>/
├── seq1/
│   └── top_activations.json
├── seq2/
│   └── top_activations.json
│   └── activation_indices.json
├── top_seq1.txt
├── top_seq2.txt
└── analysis_differential_12.txt
```

---

### `compute_activations.py`
**Purpose:** Computes Top-K activated sequences for every latent across all layers of the CLT model. Used for dataset-wide analysis. The results of this have already been saved in `top10_activations.pt` (which can be downloaded [here](https://huggingface.co/datasets/ktalreja/ProtoMechData/blob/main/top10_activations.pt)), but feel free to change it or run it again.

**Key Features:**
- Processes batches of sequences from a Parquet dataset
- Tracks the top-K sequences that maximally activate each latent
- Stores metadata (Entry Name, Sequence, activation trace) for each hit

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--checkpoint` | Path to CLT checkpoint |
| `--parquet` | Path to input Parquet file with sequences |
| `--output` | Output `.pt` file for results |
| `--k` | Number of top sequences to track per latent |

---

## Visualization & Graph Tools

### `generate_pymol_view.py`
**Purpose:** Generates PyMOL visualization scripts (`circuit_visualization.py`) to display CLT activations on 3D protein structures.

**Key Features:**
- Loads in PDB structure
- Maps sequence positions to PDB residue IDs (handles offsets)
- Generates color gradients based on activation intensity
- Supports both single-latent and full-circuit visualizations

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--clt_ckpt` | Path to CLT checkpoint |
| `--esm_path` | Path to ESM2 model |
| `--output_dir` | Output directory to leave `circuit_visualization.py` at |

**Arguments Inside Script**
| Argument | Description |
|----------|-------------|
| `PDB_NAME` | Name of PDB file |
| `SEQUENCE` | Sequence to feed in |
| `TARGET_NODES` | Dictionary containing layer:[latents] |
| `START_POS` | Starting position of sequence to annotate PDB at |
| `END_POS` | Ending position of sequence to annotate PDB at |
| `PDB_OFFSET` | Correcting for differences between PDB and sequence position numbers |

- For Kinase, we use `P83104_5_281_4o91.1.A.cif`. 
- For NAD(P)-binding, we use `Q5RKL5_26_469_7tai.1.C.cif`.
- For GB1, we use `1PGA.cif`.
- For Fig. 2B in the paper, we use `fold_gb1_mutant_model_0.cif`.

**Output:**
- `args.output_dir/circuit_visualization.py` — PyMOL script with color gradient commands

---

### `get_edge_weights.py`
**Purpose:** Computes **virtual edge weights** between circuit nodes using the CLT's local replacement model. This reveals how activations in earlier layers influence activations in later layers using a gradient-based attribution.

**Method:**
For each target node (layer L, token T, feature F), compute the influence of each source node in earlier layers using batched gradient computation.

---

## Directory Structure

```
visualization/
├── circuit_analysis_function.py  # Multi-sequence function circuit analysis
├── circuit_analysis.py           # PyMOL-integrated circuit analysis
├── compute_activations.py        # Dataset-wide Top-K tracker
├── generate_pymol_view.py        # 3D structure visualization generator
├── get_edge_weights.py           # Virtual weight computation
└── <output_directories>/         # Analysis outputs
    ├── kinase/
    ├── GB1/
    └── NAD(P)-binding/
```

---

## Output File Formats

### `activation_indices.json`
List of activated nodes with metadata:
```json
[
  [layer, token_idx, activation_value, latent_idx],
  ...
]
```

### `top_activations.json`
Structured for website integration:
```json
{
  "family": "kinase",
  "nodes": [
    {
      "layer": 2,
      "latent": 1917,
      "top_hits": [
        {"entry": "P83104", "score": 5.23, "position": 45, "context": "DFGLA"},
        ...
      ]
    }
  ]
}
```

### `virtual_weights.json` / `edge_weights.json`
Edge list for graph visualization:
```json
[
  {"src": [layer, token, latent], "tgt": [layer, token, latent], "weight": 0.0523},
  ...
]
```

---

## Website scripts

These scripts are used to facilitate running ProtoMech on [Google Colab](https://colab.research.google.com/drive/13QsDdwgKX-DWbH01qj8ZzlyY-T8MjQIx?usp=sharing).
- `auto_discover_circuit_website.py`
- `circuit_analysis_builder_website.py`