"""Training utilities for DCGAN."""

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .ddp_utils import synchronize_metrics
from .evaluation import create_fixed_noise
from .models import Discriminator, Generator
from .utils import plot_losses, save_checkpoint, save_samples
from .wandb_logging import log_step_metrics


def setup_models(config: dict, device: torch.device, rank: int) -> Tuple[nn.Module, nn.Module]:
    """Setup generator and discriminator models with DDP.
    
    Args:
        config: Configuration dictionary
        device: Training device
        rank: Process rank for DDP
        
    Returns:
        Tuple of (generator, discriminator) wrapped with DDP
    """
    # Create models
    generator = Generator(
        latent_dim=config["model"]["latent_dim"],
        channels=config["model"]["generator"]["channels"],
        output_size=config["dataset"]["image_size"],
    ).to(device)
    
    discriminator = Discriminator(
        channels=config["model"]["discriminator"]["channels"], 
        input_size=config["dataset"]["image_size"]
    ).to(device)
    
    # Wrap models with DDP
    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])
    
    return generator, discriminator


def setup_optimizers(generator: nn.Module, discriminator: nn.Module, config: dict) -> Tuple[optim.Optimizer, optim.Optimizer]:
    """Setup optimizers for generator and discriminator.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        config: Configuration dictionary
        
    Returns:
        Tuple of (g_optimizer, d_optimizer)
    """
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=config["training"]["learning_rate"]["generator"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config["training"]["learning_rate"]["discriminator"],
        betas=(config["training"]["beta1"], config["training"]["beta2"]),
    )
    
    return g_optimizer, d_optimizer


def train_discriminator_step(
    discriminator: nn.Module,
    generator: nn.Module,
    real_images: torch.Tensor,
    criterion: nn.Module,
    d_optimizer: optim.Optimizer,
    config: dict,
    device: torch.device
) -> torch.Tensor:
    """Train discriminator for one step.
    
    Args:
        discriminator: Discriminator model
        generator: Generator model  
        real_images: Real images batch
        criterion: Loss criterion
        d_optimizer: Discriminator optimizer
        config: Configuration dictionary
        device: Training device
        
    Returns:
        torch.Tensor: Discriminator loss
    """
    batch_size = real_images.size(0)
    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)
    
    d_optimizer.zero_grad()
    
    # Loss on real images
    outputs_real = discriminator(real_images)
    d_loss_real = criterion(outputs_real, real_labels)
    d_loss_real.backward()
    
    # Loss on fake images
    z = torch.randn(batch_size, config["model"]["latent_dim"], device=device)
    fake_images = generator(z)
    outputs_fake = discriminator(fake_images.detach())
    d_loss_fake = criterion(outputs_fake, fake_labels)
    d_loss_fake.backward()
    
    d_loss = d_loss_real + d_loss_fake
    d_optimizer.step()
    
    return d_loss


def train_generator_step(
    generator: nn.Module,
    discriminator: nn.Module,
    criterion: nn.Module,
    g_optimizer: optim.Optimizer,
    batch_size: int,
    config: dict,
    device: torch.device
) -> torch.Tensor:
    """Train generator for one step.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        criterion: Loss criterion
        g_optimizer: Generator optimizer
        batch_size: Size of the current batch
        config: Configuration dictionary
        device: Training device
        
    Returns:
        torch.Tensor: Generator loss
    """
    real_labels = torch.ones(batch_size, 1, device=device)
    
    g_optimizer.zero_grad()
    
    z = torch.randn(batch_size, config["model"]["latent_dim"], device=device)
    fake_images = generator(z)
    outputs = discriminator(fake_images)
    g_loss = criterion(outputs, real_labels)
    
    g_loss.backward()
    g_optimizer.step()
    
    return g_loss


def create_output_directories(config: dict):
    """Create output directories for checkpoints, samples, and logs.
    
    Args:
        config: Configuration dictionary
    """
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["sample_dir"], exist_ok=True)
    os.makedirs(config["output"]["log_dir"], exist_ok=True)


def save_final_outputs(generator: nn.Module, discriminator: nn.Module, g_optimizer: optim.Optimizer, 
                      d_optimizer: optim.Optimizer, g_losses: list, d_losses: list, config: dict):
    """Save final model checkpoint and loss plots.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        g_losses: List of generator losses
        d_losses: List of discriminator losses
        config: Configuration dictionary
    """
    save_checkpoint(
        generator.module,
        discriminator.module,
        g_optimizer,
        d_optimizer,
        config["training"]["num_epochs"],
        config["output"]["checkpoint_dir"],
    )
    plot_losses(g_losses, d_losses, os.path.join(config["output"]["log_dir"], "final_losses.png"))