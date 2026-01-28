# Training

This folder contains the training code for the Per-Layer Transcoder (PLT) model, a sparse autoencoder that replaces ESM-2 MLP blocks with cross-layer reconstruction capabilities.

## Key Components

- `plt_model.py`: PLT model architecture with top-k sparse activation and cross-layer decoding
- `plt_module.py`: PyTorch Lightning training module with loss functions and optimization
- `run_plt.py`: Main training script with argument parsing and logging
- `main.sh`: Shell script for running training with default parameters

## Usage

Run training with default settings:

```bash
./main_plt.sh
```

Or customize parameters by changing variables in main_plt.sh. You can also find the trained PLT
at: https://huggingface.co/anonymous-hf-user/ProtoMechModels/tree/main/PLT_L6_D3200