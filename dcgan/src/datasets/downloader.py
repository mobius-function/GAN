"""Dataset downloading utilities for fetching images from HuggingFace."""

from typing import List, Optional, Tuple

import requests
from tqdm import tqdm

from .cache_manager import CacheManager


class HuggingFaceDownloader:
    """Downloads celebrity dataset from HuggingFace."""
    
    BASE_URL = "https://datasets-server.huggingface.co/rows"
    DATASET_NAME = "ares1123/celebrity_dataset"
    
    def __init__(self, cache_manager: CacheManager):
        """Initialize downloader with cache manager.
        
        Args:
            cache_manager: CacheManager instance for handling caching
        """
        self.cache_manager = cache_manager
    
    def check_dataset_availability(self) -> bool:
        """Check if HuggingFace dataset is accessible.
        
        Returns:
            True if dataset is accessible, False otherwise
        """
        try:
            url = f"{self.BASE_URL}?dataset={self.DATASET_NAME}&config=default&split=train&offset=0&length=1"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return len(data.get("rows", [])) > 0
        except Exception as e:
            print(f"Error checking HF dataset: {e}")
            return False
    
    def fetch_image_urls(self, max_images: Optional[int] = None) -> List[str]:
        """Fetch image URLs from HuggingFace API.
        
        Args:
            max_images: Maximum number of image URLs to fetch
            
        Returns:
            List of image URLs
        """
        all_urls = []
        offset = 0
        batch_size = 100
        target_count = max_images if max_images else float('inf')
        
        print("Fetching image URLs from HuggingFace...")
        
        while len(all_urls) < target_count:
            url = f"{self.BASE_URL}?dataset={self.DATASET_NAME}&config=default&split=train&offset={offset}&length={batch_size}"
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("rows"):
                    break
                
                for row in data["rows"]:
                    img_url = row["row"]["image"]["src"]
                    all_urls.append(img_url)
                    
                    if len(all_urls) >= target_count:
                        break
                
                offset += batch_size
                
                if len(data["rows"]) < batch_size:
                    break
                    
            except Exception as e:
                print(f"Error fetching URLs at offset {offset}: {e}")
                break
        
        # Sort for deterministic ordering
        all_urls.sort()
        if max_images:
            all_urls = all_urls[:max_images]
        
        print(f"Fetched {len(all_urls)} URLs")
        return all_urls
    
    def download_and_cache_images(self, urls: List[str], max_failure_rate: float = 0.1) -> List[Tuple[str, str]]:
        """Download and cache images from URLs.
        
        Args:
            urls: List of image URLs to download
            max_failure_rate: Maximum acceptable failure rate
            
        Returns:
            List of (url, cache_path) tuples for successfully downloaded images
            
        Raises:
            RuntimeError: If too many downloads fail
        """
        print(f"Downloading {len(urls)} images...")
        
        valid_data = []
        failed_count = 0
        
        for url in tqdm(urls, desc="Downloading images"):
            cache_path = self.cache_manager.get_image_cache_path(url)
            
            # Check if already cached and valid
            if self.cache_manager.validate_cached_image(cache_path):
                valid_data.append((url, cache_path))
            else:
                # Download and cache
                img = self.cache_manager.download_and_cache_image(url, cache_path)
                if img:
                    valid_data.append((url, cache_path))
                else:
                    failed_count += 1
                    if failed_count > len(urls) * max_failure_rate:
                        raise RuntimeError(f"Too many download failures: {failed_count}/{len(urls)}")
        
        print(f"Successfully downloaded {len(valid_data)} images ({failed_count} failures)")
        return valid_data