#!/bin/bash

# Simple DDP training launcher
echo "Starting DDP training with all available GPUs..."
python train_ddp.py --config config.yaml