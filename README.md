# GAN Implementation for CelebA Dataset

This is a basic GAN (Generative Adversarial Network) implementation for generating face images using the CelebA dataset.

## Project Structure

- `config.yaml` - Default configuration file with hyperparameters
- `models.py` - Generator and Discriminator neural network architectures
- `dataset.py` - CelebA dataset loader and preprocessing
- `utils.py` - Utility functions for visualization and checkpointing
- `train.py` - Main training script
- `requirements.txt` - Python package dependencies

## Setup

1. Install dependencies using one of these methods:

   ```bash
   # Using uv (if available)
   uv pip install -r requirements.txt
   
   # Using pip
   pip install -r requirements.txt
   
   # Using python3 directly
   python3 -m pip install -r requirements.txt
   ```

2. The CelebA dataset will be automatically downloaded on first run.

## Usage

### Training with default configuration:
```bash
python train.py
```

### Training with custom configuration:
```bash
python train.py --config my_config.yaml
```

## Configuration

The `config.yaml` file contains all hyperparameters including:
- Dataset settings (batch size, image size)
- Model architecture (latent dimension, channel sizes)
- Training parameters (learning rates, epochs)
- Output directories for samples and checkpoints

## Output

- Generated samples: `./samples/`
- Model checkpoints: `./checkpoints/`
- Training loss plots: `./logs/`

## Features

- Configurable model architecture
- Automatic checkpointing
- Sample generation during training
- Loss visualization
- Resume training from checkpoints