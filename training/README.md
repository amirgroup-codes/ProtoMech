# Training

This folder contains the training code for the Cross-Layer Transcoder (CLT) model, a sparse autoencoder that replaces ESM-2 MLP blocks with cross-layer reconstruction capabilities.

## Key Components

- `clt_model.py`: CLT model architecture with top-k sparse activation and cross-layer decoding
- `clt_module.py`: PyTorch Lightning training module with loss functions and optimization
- `data_module.py`: Data loading and preprocessing for protein sequences
- `run_clt.py`: Main training script with argument parsing and logging
- `main.sh`: Shell script for running training with default parameters

## Usage

Run training with default settings:

```bash
./main.sh
```

Or customize parameters by changing variables in main.sh. You can also find the trained CLT
at: https://huggingface.co/anonymous-hf-user/ProtoMechModels/tree/main/CLT_L6_D3200