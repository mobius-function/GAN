# Datasets Module

This module provides modular components for dataset handling in DCGAN training.

## Structure

- `__init__.py` - Package initialization and exports
- `celebrity.py` - Main CelebrityFacesDataset implementation
- `cache_manager.py` - Cache management utilities for efficient data storage
- `downloader.py` - HuggingFace dataset downloading utilities

## Key Components

### CacheManager
Handles all caching operations including:
- Image cache path generation using URL hashing
- Cache validation and integrity checking
- Metadata and URL persistence
- Download and cache operations

### HuggingFaceDownloader
Manages dataset downloading from HuggingFace:
- Dataset availability checking
- Batch URL fetching from API
- Parallel image downloading with progress tracking
- Failure rate monitoring

### CelebrityFacesDataset
PyTorch Dataset implementation with:
- Automatic caching and validation
- Train/validation splitting with fixed seed
- Transform pipeline support
- Fallback mechanisms for corrupted images

## Usage

```python
from src.datasets import CelebrityFacesDataset, get_dataloader, get_dataloader_ddp

# For single GPU training
dataloader = get_dataloader(config)

# For distributed training
dataloader, sampler = get_dataloader_ddp(config, rank, world_size, split="train")
```

## Configuration

The dataset accepts the following configuration parameters:
- `image_size`: Target image size for resizing
- `batch_size`: Batch size for DataLoader
- `max_images`: Maximum number of images to use (None for all)
- `train_ratio`: Train/validation split ratio (default: 0.8)
- `dataset_seed`: Seed for reproducible splits (default: 42)
- `cache_dir`: Directory for caching (default: "./cache")
- `validate_cache`: Whether to validate cached images (default: True)
- `num_workers`: Number of DataLoader workers
- `deterministic`: Disable shuffling for deterministic behavior