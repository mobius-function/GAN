"""Evaluation utilities for DCGAN training with IS and FID metrics."""

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from scipy import linalg


def evaluate_generator(
    generator: nn.Module, 
    fixed_noise: torch.Tensor, 
    epoch: int, 
    sample_dir: str, 
    device: torch.device,
    nrow: int = 4
) -> torch.Tensor:
    """Evaluate generator by creating sample images for visual inspection.
    
    Args:
        generator: Generator model
        fixed_noise: Fixed noise vectors for consistent comparison
        epoch: Current epoch number
        sample_dir: Directory to save sample images
        device: Device to run generation on
        nrow: Number of images per row in saved grid
        
    Returns:
        torch.Tensor: Generated images moved to CPU
    """
    generator.eval()
    
    with torch.no_grad():
        # Generate sample images from fixed noise
        generated_images = generator(fixed_noise)
        
        # Save samples for visual inspection
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, f"generated_epoch_{epoch}.png")
        vutils.save_image(
            generated_images, 
            sample_path, 
            normalize=True, 
            nrow=nrow
        )
    
    generator.train()
    return generated_images.detach().cpu()


def create_fixed_noise(batch_size: int, latent_dim: int, device: torch.device, seed: int = 42) -> torch.Tensor:
    """Create fixed noise vectors for consistent evaluation across epochs.
    
    Uses a fixed seed to ensure the same noise vectors are generated every time,
    making evaluation results reproducible across different runs.
    
    Args:
        batch_size: Number of images to generate
        latent_dim: Latent dimension size
        device: Device to create noise on
        seed: Random seed for reproducible noise generation
        
    Returns:
        torch.Tensor: Fixed noise vectors (same values every time)
    """
    # Save current random state
    current_state = torch.get_rng_state()
    
    # Set fixed seed for reproducible noise
    torch.manual_seed(seed)
    
    # Generate fixed noise
    fixed_noise = torch.randn(batch_size, latent_dim, device=device)
    
    # Restore original random state (so training randomness is not affected)
    torch.set_rng_state(current_state)
    
    return fixed_noise


class InceptionV3FeatureExtractor(nn.Module):
    """Inception V3 model for feature extraction used in FID and IS calculation."""
    
    def __init__(self):
        super().__init__()
        self.model = inception_v3(pretrained=True)
        self.model.eval()
        
        # Remove the final classification layer
        self.model.fc = nn.Identity()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """Extract features from Inception V3.
        
        Args:
            x: Input images tensor [B, 3, 299, 299]
            
        Returns:
            torch.Tensor: Features [B, 2048]
        """
        return self.model(x)


def preprocess_images_for_inception(images: torch.Tensor) -> torch.Tensor:
    """Preprocess images for Inception V3 (requires 299x299, normalized).
    
    Args:
        images: Images tensor [B, 3, H, W] in range [-1, 1]
        
    Returns:
        torch.Tensor: Preprocessed images [B, 3, 299, 299] in range [0, 1]
    """
    # Convert from [-1, 1] to [0, 1]
    images = (images + 1) / 2.0
    
    # Resize to 299x299 for Inception V3
    if images.shape[-1] != 299:
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    # Normalize with ImageNet statistics
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images = normalize(images)
    
    return images


def calculate_inception_score(images: torch.Tensor, batch_size: int = 32, splits: int = 10) -> Tuple[float, float]:
    """Calculate Inception Score (IS) for generated images.
    
    Args:
        images: Generated images tensor [N, 3, H, W] in range [-1, 1]
        batch_size: Batch size for processing
        splits: Number of splits for calculating standard deviation
        
    Returns:
        Tuple of (mean_IS, std_IS)
    """
    device = images.device
    N = images.shape[0]
    
    # Load Inception model
    inception_model = InceptionV3FeatureExtractor().to(device)
    
    # Preprocess images
    images = preprocess_images_for_inception(images)
    
    # Get predictions in batches
    preds = []
    for i in range(0, N, batch_size):
        batch = images[i:i+batch_size]
        with torch.no_grad():
            # Get logits from inception (before softmax)
            logits = inception_model(batch)
            # Convert to probabilities
            pred = F.softmax(logits, dim=1)
            preds.append(pred)
    
    preds = torch.cat(preds, dim=0)
    
    # Calculate IS for each split
    scores = []
    for i in range(splits):
        part = preds[i * (N // splits):(i + 1) * (N // splits)]
        
        # Calculate marginal distribution p(y)
        p_y = torch.mean(part, dim=0)
        
        # Calculate KL divergence for each sample
        kl_div = part * (torch.log(part + 1e-8) - torch.log(p_y + 1e-8))
        kl_div = torch.sum(kl_div, dim=1)
        
        # IS = exp(E[KL(p(y|x) || p(y))])
        is_score = torch.exp(torch.mean(kl_div))
        scores.append(is_score.cpu().numpy())
    
    return np.mean(scores), np.std(scores)


def calculate_fid_score(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Calculate Fréchet Inception Distance (FID) between real and fake images.
    
    Args:
        real_features: Features from real images [N, 2048]
        fake_features: Features from fake images [N, 2048]
        
    Returns:
        float: FID score (lower is better)
    """
    # Calculate mean and covariance for real and fake features
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_fake
    covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
    
    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid)


def extract_inception_features(images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
    """Extract features from images using Inception V3.
    
    Args:
        images: Images tensor [N, 3, H, W] in range [-1, 1]
        batch_size: Batch size for processing
        
    Returns:
        np.ndarray: Features [N, 2048]
    """
    device = images.device
    N = images.shape[0]
    
    # Load Inception model
    inception_model = InceptionV3FeatureExtractor().to(device)
    
    # Preprocess images
    images = preprocess_images_for_inception(images)
    
    # Extract features in batches
    features = []
    for i in range(0, N, batch_size):
        batch = images[i:i+batch_size]
        with torch.no_grad():
            feat = inception_model(batch)
            features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def evaluate_gan_metrics(
    generator: nn.Module,
    real_dataloader: DataLoader,
    device: torch.device,
    latent_dim: int,
    num_samples: int = 10000,
    batch_size: int = 64
) -> Tuple[float, float, float, int]:
    """Comprehensive GAN evaluation with IS and FID scores.
    
    Args:
        generator: Generator model
        real_dataloader: DataLoader with real images
        device: Device to run evaluation on
        latent_dim: Latent dimension of the generator
        num_samples: Number of samples to generate for evaluation
        batch_size: Batch size for generation and feature extraction
        
    Returns:
        Tuple of (IS_mean, IS_std, FID_score, num_real_samples_used)
    """
    generator.eval()
    
    print(f"Evaluating GAN with {num_samples} generated samples...")
    
    # Generate fake images
    fake_images = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            z = torch.randn(current_batch_size, latent_dim, device=device)
            fake_batch = generator(z)
            fake_images.append(fake_batch)
    
    fake_images = torch.cat(fake_images, dim=0)
    
    # Calculate Inception Score
    print("Calculating Inception Score...")
    is_mean, is_std = calculate_inception_score(fake_images)
    
    # Extract features from fake images
    print("Extracting features from generated images...")
    fake_features = extract_inception_features(fake_images)
    
    # Extract features from real images
    print("Extracting features from real images...")
    real_features = []
    real_count = 0
    
    for real_batch, _ in real_dataloader:
        if real_count >= num_samples:
            break
        
        real_batch = real_batch.to(device)
        current_size = min(real_batch.size(0), num_samples - real_count)
        real_batch = real_batch[:current_size]
        
        real_feat = extract_inception_features(real_batch)
        real_features.append(real_feat)
        real_count += current_size
    
    real_features = np.concatenate(real_features, axis=0)
    
    # Calculate FID
    print("Calculating FID score...")
    fid_score = calculate_fid_score(real_features, fake_features)
    
    generator.train()
    
    return is_mean, is_std, fid_score, real_count