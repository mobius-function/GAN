"""Cache management utilities for dataset operations."""

import hashlib
import pickle
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from tqdm import tqdm


class CacheManager:
    """Manages caching for dataset images and metadata."""
    
    def __init__(self, cache_dir: str = "./cache", max_images: Optional[int] = None, 
                 dataset_seed: int = 42):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
            max_images: Maximum number of images to cache
            dataset_seed: Seed for reproducible dataset generation
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_images = max_images
        self.dataset_seed = dataset_seed
        
        # Setup cache paths
        self.images_cache_dir = self.cache_dir / "images"
        self.images_cache_dir.mkdir(exist_ok=True)
        self.urls_cache_file = self.cache_dir / f"celebrity_urls_{max_images}_{dataset_seed}.pkl"
        self.metadata_file = self.cache_dir / f"metadata_{max_images}_{dataset_seed}.pkl"
    
    def get_image_cache_path(self, url: str) -> Path:
        """Generate cache path for image based on URL hash.
        
        Args:
            url: Image URL
            
        Returns:
            Path to cached image file
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.images_cache_dir / f"{url_hash}.jpg"
    
    def validate_cached_image(self, cache_path: Path) -> bool:
        """Validate that cached image is not corrupted.
        
        Args:
            cache_path: Path to cached image
            
        Returns:
            True if image is valid, False otherwise
        """
        if not cache_path.exists():
            return False
        try:
            img = Image.open(cache_path)
            img.verify()
            return True
        except Exception:
            return False
    
    def download_and_cache_image(self, url: str, cache_path: Path) -> Optional[Image.Image]:
        """Download image and save to cache.
        
        Args:
            url: Image URL to download
            cache_path: Path where to save the image
            
        Returns:
            Downloaded image or None if failed
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Validate and convert image
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Save to cache
            img.save(cache_path, "JPEG", quality=95)
            return img
            
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return None
    
    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to cache.
        
        Args:
            metadata: Metadata dictionary to save
        """
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load metadata from cache.
        
        Returns:
            Metadata dictionary or None if not found
        """
        if not self.metadata_file.exists():
            return None
        
        try:
            with open(self.metadata_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None
    
    def save_urls(self, urls: List[str]) -> None:
        """Save URLs list to cache.
        
        Args:
            urls: List of URLs to save
        """
        with open(self.urls_cache_file, 'wb') as f:
            pickle.dump(urls, f)
    
    def load_urls(self) -> Optional[List[str]]:
        """Load URLs from cache.
        
        Returns:
            List of URLs or None if not found
        """
        if not self.urls_cache_file.exists():
            return None
        
        try:
            with open(self.urls_cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading URLs: {e}")
            return None
    
    def validate_cache(self, urls: List[str], validation_threshold: float = 0.95) -> Tuple[bool, List[Tuple[str, Path]]]:
        """Validate cached images.
        
        Args:
            urls: List of URLs to validate
            validation_threshold: Minimum percentage of valid images required
            
        Returns:
            Tuple of (is_valid, valid_data) where valid_data is list of (url, cache_path) tuples
        """
        print("Validating cached images...")
        valid_data = []
        
        for url in tqdm(urls, desc="Validating cache"):
            cache_path = self.get_image_cache_path(url)
            if self.validate_cached_image(cache_path):
                valid_data.append((url, cache_path))
            else:
                # Try to re-download invalid image
                img = self.download_and_cache_image(url, cache_path)
                if img:
                    valid_data.append((url, cache_path))
        
        is_valid = len(valid_data) >= len(urls) * validation_threshold
        if not is_valid:
            print(f"Too many invalid cached images ({len(valid_data)}/{len(urls)})")
        
        return is_valid, valid_data