# ProtoMech
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="ProtoMech_Logo_Dark.svg">
    <img alt="ProtoMech Logo" src="ProtoMech_Logo_Light.svg" width="60%">
  </picture>
</p>

This is the official code repository for the paper "Protein Circuit Tracing via Cross-layer Transcoders", by Darin Tsui, Kunal Talreja, Daniel Saeedi, and Amirali Aghazadeh. A link to the paper can be found [here](https://arxiv.org/abs/XXXX.XXXXX). 

Additionally, one can explore protein circuits through our [web-based visualizer](https://protmech.github.io/)!

---

## Quick Start

The easiest way to get started with ProtoMech is through our interactive [Google Colab notebook](https://colab.research.google.com/github/amirgroup-codes/ProtoMech/blob/main/ProtoMech.ipynb). No local installation is required.

### Workflow 

1. **Circuit Discovery** (optional): Train a probe on your custom dataset (Binary classification or Regression) to identify circuits.
2. **Interactive Visualization**: Generate files required for our [website](https://protmech.github.io/) and visualize circuits!

If you skip step 1, you can obtain circuit files in two ways:
- **Use Our Pre-discovered Library**: If you want to explore circuits from our paper, we provide a curated list of circuits [here](https://github.com/amirgroup-codes/ProtoMech/blob/main/visualization/circuits.md) you can access through our notebook.
- **Auto-generate Your Own**: Even without a custom dataset, you can still generate a circuit! Just leave the `circuit` option blank. 

## Environment Setup

Create the conda environment by running:

```bash
conda env create -f clt.yml
conda activate clt
```

## Repository Structure

```
ProtoMech/
├── training/              # CLT training code
├── training_transcoder/   # PLT training code
├── circuit_utils/         # Core circuit discovery utilities
├── family_circuit/        # Protein family-based circuit discovery
├── function_circuit/      # DMS function-based circuit discovery
├── steering/              # Probe and DMS steering experiments
├── esm_steering/          # CAA (Contrastive Activation Addition) steering
├── visualization/         # Circuit analysis and PyMOL visualization
├── data/                  # Training data generation
└── plots_and_tables/      # Plots and tables
```

---

## Model Architectures

### Cross-Layer Transcoder (CLT)
**Location**: `training/clt_model.py`

Replaces ESM-2 MLP blocks with a sparse transcoder using information from *all* preceding layers:
- **Top-K Activation**: Only top-k latents are active (enforces sparsity)
- **Cross-Layer Decoding**: Layer *l* reconstructs using latents from layers 0 to *l*
- **AuxK Loss**: Encourages rarely-used latents to activate

### Per-Layer Transcoder (PLT)
**Location**: `training_transcoder/plt_model.py`

Baseline where each layer has independent encoder/decoder pairs. Layer *l* only uses its own latents.

---

## Training

```bash
# Train CLT
cd training && sh main.sh

# Train PLT
cd training_transcoder && sh main_plt.sh
```

If you would like to train your own model, download `training_sequences_5m.a2m` from [https://huggingface.co/datasets/ktalreja/ProtoMechData](https://huggingface.co/datasets/ktalreja/ProtoMechData) and put it in the `data` folder. 

---

## Circuit Discovery

Identifies minimal subsets of latents that recover a target property (family classification or DMS fitness).

### Core Utilities (`circuit_utils/`)

| File | Description |
|------|-------------|
| `clt_circuit.py` | CLT circuit discovery |
| `plt_circuit.py` | PLT circuit discovery |
| `esm_activation.py` | ESM-2 activation extraction |

### Family Circuit Discovery (`family_circuit/`)

Discovers circuits distinguishing protein families (InterPro domains).

```bash
cd family_circuit
sh main.sh                        # Full run for all families
sh main.sh --target IPR000724     # Specific family
```

You can download our Swiss-Prot data used for our family circuits, `swissprot_seqid30_75k_all_info_with_3di.parquet`, from [https://huggingface.co/datasets/ktalreja/ProtoMechData](https://huggingface.co/datasets/ktalreja/ProtoMechData) and put it in the `data` folder. 

### Function Circuit Discovery (`function_circuit/`)

Discovers circuits using DMS fitness data.

```bash
cd function_circuit && sh main.sh
```

---

## Steering

Modifies sequence generation by amplifying or ablating circuit nodes.

### Replacement Models (`steering/`)

| File | Description |
|------|-------------|
| `full_replacement_models.py` | `FullCLTReplacementModel`, `FullPLTReplacementModel` |
| `local_replacement_models.py` | Local replacement model for CLT |
| `run_probe_steering.py` | Probe-based steering |

### CAA Steering (`esm_steering/`)

Contrastive Activation Addition steering using steering vectors from contrastive pairs.

```bash
cd esm_steering && sh main_caa_steering.sh
```

---

## Visualization

**Location**: `visualization/`

| File | Description |
|------|-------------|
| `circuit_analysis.py` | Family-level circuit analysis |
| `circuit_analysis_function.py` | Function/DMS-level analysis |
| `generate_pymol_view.py` | PyMOL visualization scripts |
| `compute_activations.py` | Computes top-10 sequences per act |

If you want to use `compute_activations.py` instead of using the pre-saved top activation results found in `top10_activations.pt` (which can be found [here](https://huggingface.co/datasets/ktalreja/ProtoMechData/blob/main/top10_activations.pt)), download `swissprot_full.parquet` from [https://huggingface.co/datasets/ktalreja/ProtoMechData](https://huggingface.co/datasets/ktalreja/ProtoMechData) and put it in the `data` folder.

---

## Previous Data

You can find the models at [https://huggingface.co/ktalreja/ProtoMechModels](https://huggingface.co/ktalreja/ProtoMechModels) and the data used in this paper at [https://huggingface.co/datasets/ktalreja/ProtoMechData](https://huggingface.co/datasets/ktalreja/ProtoMechData).
