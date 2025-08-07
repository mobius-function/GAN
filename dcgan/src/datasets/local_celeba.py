"""Local CelebA Dataset implementation."""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


class LocalCelebADataset(Dataset):
    """Local CelebA Dataset from extracted archive."""

    def __init__(
        self,
        data_dir: str,
        transform=None,
        max_images: Optional[int] = None,
    ):
        """Initialize local CelebA dataset.
        
        Args:
            data_dir: Directory containing the images
            transform: Torchvision transforms to apply
            max_images: Maximum number of images to use (None for all)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(self.data_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        # Limit number of images if specified
        if max_images and max_images < len(self.image_files):
            self.image_files = self.image_files[:max_images]
        
        print(f"Loaded {len(self.image_files)} images from {data_dir}")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get dataset item.
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Tuple of (image_tensor, label) where label is always 0
        """
        img_path = self.data_dir / self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            return image, 0
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            
            # Return black image as fallback
            image = Image.new("RGB", (64, 64), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, 0


def get_transform(image_size: int) -> transforms.Compose:
    """Get standard transform for CelebA dataset.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_dataloader(config: dict) -> DataLoader:
    """Create DataLoader for local CelebA dataset.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataLoader instance
    """
    transform = get_transform(config["dataset"]["image_size"])
    
    # Create dataset
    dataset = LocalCelebADataset(
        data_dir=config["dataset"]["data_dir"],
        transform=transform,
        max_images=config["dataset"].get("max_images", None),
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        drop_last=True,
        pin_memory=True if config.get("device") == "cuda" else False
    )
    
    return dataloader


def get_dataloader_ddp(config: dict, rank: int, world_size: int) -> Tuple[DataLoader, DistributedSampler]:
    """Get dataloader with DistributedSampler for DDP.
    
    Args:
        config: Configuration dictionary
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        Tuple of (DataLoader, DistributedSampler)
    """
    transform = get_transform(config["dataset"]["image_size"])
    
    # Create dataset
    dataset = LocalCelebADataset(
        data_dir=config["dataset"]["data_dir"],
        transform=transform,
        max_images=config["dataset"].get("max_images", None),
    )
    
    # Create distributed sampler
    deterministic = config["dataset"].get("deterministic", False)
    should_shuffle = not deterministic
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=should_shuffle
    )
    
    # Adjust batch size per GPU
    batch_size_per_gpu = config["dataset"]["batch_size"] // world_size
    print(f"Available GPUs: {world_size}, Workers per GPU: {config['dataset']['num_workers']}, Batch size per GPU: {batch_size_per_gpu}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader, sampler