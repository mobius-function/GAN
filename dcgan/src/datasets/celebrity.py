"""Celebrity Faces Dataset implementation with caching support."""

from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from .cache_manager import CacheManager
from .downloader import HuggingFaceDownloader


class CelebrityFacesDataset(Dataset):
    """Celebrity Faces Dataset from Hugging Face with advanced caching."""

    def __init__(
        self,
        transform=None,
        max_images: Optional[int] = None,
        dataset_seed: int = 42,
        cache_dir: str = "./cache",
        validate_cache: bool = True
    ):
        """Initialize celebrity faces dataset.
        
        Args:
            transform: Torchvision transforms to apply
            max_images: Maximum number of images to use
            dataset_seed: Seed for reproducible dataset generation
            cache_dir: Directory for caching images
            validate_cache: Whether to validate cached images
        """
        self.transform = transform
        self.dataset_seed = dataset_seed
        self.validate_cache = validate_cache
        
        # Initialize cache manager and downloader
        self.cache_manager = CacheManager(cache_dir, max_images, dataset_seed)
        self.downloader = HuggingFaceDownloader(self.cache_manager)
        
        print("Initializing celebrity faces dataset...")
        
        # Load or download dataset
        self._load_or_download_dataset()
        
        # Use all data for training (no splits needed for GANs)
        self.image_data = self.all_image_data.copy()
        
        print(f"Dataset ready: {len(self.image_data)} images for training")
    
    def _load_or_download_dataset(self):
        """Load dataset from cache or download if needed."""
        # Try to load from cache first
        if self._load_from_cache():
            print("Successfully loaded dataset from cache")
            return
        
        # Download if cache is incomplete or invalid
        print("Cache incomplete or invalid, downloading dataset...")
        self._download_and_cache_dataset()
    
    def _load_from_cache(self) -> bool:
        """Try to load dataset from cache.
        
        Returns:
            True if successfully loaded from cache, False otherwise
        """
        # Load metadata
        metadata = self.cache_manager.load_metadata()
        if not metadata:
            return False
        
        # Check if cache matches current parameters
        if (metadata.get('max_images') != self.cache_manager.max_images or 
            metadata.get('dataset_seed') != self.cache_manager.dataset_seed):
            print("Cache parameters don't match, need to rebuild")
            return False
        
        # Load URLs
        urls = self.cache_manager.load_urls()
        if not urls:
            return False
        
        self.all_image_urls = urls
        
        # Validate cached images if requested
        if self.validate_cache:
            is_valid, valid_data = self.cache_manager.validate_cache(urls)
            if not is_valid:
                return False
            self.all_image_data = valid_data
        else:
            # Trust cache without validation
            self.all_image_data = [
                (url, self.cache_manager.get_image_cache_path(url)) 
                for url in urls
            ]
        
        print(f"Loaded {len(self.all_image_data)} images from cache")
        return True
    
    def _download_and_cache_dataset(self):
        """Download dataset from HuggingFace and cache images."""
        # Check dataset availability
        if not self.downloader.check_dataset_availability():
            raise RuntimeError("Cannot access HuggingFace celebrity dataset")
        
        # Fetch URLs
        urls = self.downloader.fetch_image_urls(self.cache_manager.max_images)
        
        # Download and cache images
        valid_data = self.downloader.download_and_cache_images(urls)
        
        # Save to cache
        self.all_image_urls = [url for url, _ in valid_data]
        self.all_image_data = valid_data
        
        # Save URLs and metadata
        self.cache_manager.save_urls(self.all_image_urls)
        metadata = {
            'max_images': self.cache_manager.max_images,
            'dataset_seed': self.cache_manager.dataset_seed,
            'total_images': len(self.all_image_urls)
        }
        self.cache_manager.save_metadata(metadata)
        
        print(f"Saved dataset to cache: {len(self.all_image_urls)} images")
    
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get dataset item.
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Tuple of (image_tensor, label) where label is always 0
        """
        url, cache_path = self.image_data[idx]
        
        try:
            # Load from cache
            image = Image.open(cache_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            return image, 0
            
        except Exception as e:
            print(f"Error loading cached image {cache_path}: {e}")
            
            # Try to re-download
            try:
                img = self.cache_manager.download_and_cache_image(url, cache_path)
                if img:
                    if self.transform:
                        img = self.transform(img)
                    return img, 0
            except Exception:
                pass
            
            # Return black image as fallback
            image = Image.new("RGB", (256, 256), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, 0


def get_transform(image_size: int) -> transforms.Compose:
    """Get standard transform for celebrity dataset.
    
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
    """Create DataLoader for Celebrity Faces dataset.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataLoader instance
    """
    transform = get_transform(config["dataset"]["image_size"])
    
    # Create dataset with caching
    dataset = CelebrityFacesDataset(
        transform=transform,
        max_images=config["dataset"].get("max_images", None),
        dataset_seed=config["dataset"].get("dataset_seed", 42),
        cache_dir=config["dataset"].get("cache_dir", "./cache"),
        validate_cache=config["dataset"].get("validate_cache", True)
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
    
    if config["dataset"]["name"] == "celebrity":
        # Use celebrity dataset from Hugging Face - use all data for training
        dataset = CelebrityFacesDataset(
            transform=transform,
            max_images=config["dataset"].get("max_images", None),
            dataset_seed=config["dataset"].get("dataset_seed", 42),
            cache_dir=config["dataset"].get("cache_dir", "./cache"),
            validate_cache=config["dataset"].get("validate_cache", True)
        )
    else:
        # Support for other datasets (e.g., CelebA)
        from torchvision.datasets import CelebA
        
        dataset = CelebA(
            root=config["dataset"]["data_dir"],
            split="train" if split == "train" else "valid",
            transform=transform,
            download=False
        )
    
    # Create distributed sampler
    deterministic = config["dataset"].get("deterministic", False)
    should_shuffle = not deterministic  # Shuffle unless deterministic mode is enabled
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=should_shuffle
    )
    
    # Adjust batch size per GPU
    batch_size_per_gpu = config["dataset"]["batch_size"] // world_size
    
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