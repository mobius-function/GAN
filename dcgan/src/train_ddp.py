"""Simplified distributed DCGAN training script using functional approach."""

import argparse
import sys
import os

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import get_dataloader_ddp
from src.ddp_utils import cleanup_ddp, setup_ddp, setup_device, synchronize_metrics
from src.evaluation import create_fixed_noise, evaluate_generator, evaluate_gan_metrics
from src.training_utils import (
    create_output_directories, 
    save_final_outputs,
    setup_models, 
    setup_optimizers, 
    train_discriminator_step, 
    train_generator_step
)
from src.utils import load_config, plot_losses, save_checkpoint, save_samples, set_seed
from src.wandb_logging import finish_wandb, log_epoch_metrics, log_evaluation_metrics, log_generated_images, log_step_metrics, setup_wandb


def train_epoch(generator, discriminator, g_optimizer, d_optimizer, criterion, dataloader, 
               sampler, epoch, config, device, rank, use_wandb, fixed_noise, g_losses, d_losses):
    """Train for one epoch."""
    # Set epoch for distributed sampler
    sampler.set_epoch(epoch)
    
    epoch_g_loss = 0.0
    epoch_d_loss = 0.0
    
    # Progress bar only on rank 0
    if rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
    else:
        progress_bar = dataloader
    
    for i, (real_images, _) in enumerate(progress_bar):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        
        # Train Discriminator
        d_loss = train_discriminator_step(
            discriminator, generator, real_images, criterion, 
            d_optimizer, config, device
        )
        
        # Train Generator  
        g_loss = train_generator_step(
            generator, discriminator, criterion, g_optimizer,
            batch_size, config, device
        )
        
        # Accumulate losses
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        
        # Logging (only rank 0)
        if rank == 0:
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            # Update progress bar and log metrics
            if (i + 1) % config["training"]["log_interval"] == 0:
                progress_bar.set_postfix({
                    "D Loss": f"{d_loss.item():.4f}", 
                    "G Loss": f"{g_loss.item():.4f}"
                })
                
                # Log to wandb
                if use_wandb:
                    steps_per_epoch = len(dataloader)
                    log_step_metrics(d_loss, g_loss, epoch, i, steps_per_epoch)
            
            # Save samples periodically
            if (epoch * len(dataloader) + i) % config["training"]["sample_interval"] == 0:
                save_samples(generator.module, fixed_noise, epoch, config["output"]["sample_dir"], device)
    
    return epoch_d_loss / len(dataloader), epoch_g_loss / len(dataloader)


def train_gan_ddp(rank: int, world_size: int, config: dict):
    """Main training function for DDP."""
    # Setup distributed training
    setup_ddp(rank, world_size)
    
    # Set seed for reproducibility
    set_seed(config["seed"] + rank)
    
    # Set device for this process
    device = setup_device(rank)
    
    # Setup wandb logging (only on rank 0)
    use_wandb = setup_wandb(config, rank)
    if rank == 0:
        print(f"Using {world_size} GPUs for training")
    
    # Get dataloader
    train_dataloader, train_sampler = get_dataloader_ddp(config, rank, world_size)
    if rank == 0:
        print(f"Loaded dataset with {len(train_dataloader)} batches per GPU")
    
    # Setup models and optimizers
    generator, discriminator = setup_models(config, device, rank)
    g_optimizer, d_optimizer = setup_optimizers(generator, discriminator, config)
    criterion = nn.BCELoss()
    
    # Create fixed noise for evaluation
    fixed_noise = create_fixed_noise(8, config["model"]["latent_dim"], device)
    
    # Create output directories (only rank 0)
    if rank == 0:
        create_output_directories(config)
        print("Starting training...")
    
    # Training tracking
    g_losses = []
    d_losses = []
    
    # Training loop
    for epoch in range(config["training"]["num_epochs"]):
        # Train for one epoch
        avg_d_loss, avg_g_loss = train_epoch(
            generator, discriminator, g_optimizer, d_optimizer, criterion,
            train_dataloader, train_sampler, epoch, config, device, rank,
            use_wandb, fixed_noise, g_losses, d_losses
        )
        
        # Synchronize metrics across GPUs
        metrics = {"d_loss": avg_d_loss, "g_loss": avg_g_loss}
        sync_metrics = synchronize_metrics(metrics, world_size, device)
        sync_d_loss, sync_g_loss = sync_metrics["d_loss"], sync_metrics["g_loss"]
        
        if rank == 0:
            print(f"Epoch [{epoch + 1}/{config['training']['num_epochs']}] "
                  f"D Loss: {sync_d_loss:.4f}, G Loss: {sync_g_loss:.4f}")
            
            # Evaluate generator
            generated_images = evaluate_generator(
                generator.module, fixed_noise, epoch + 1, 
                config["output"]["sample_dir"], device
            )
            
            # Log to wandb
            if use_wandb:
                log_epoch_metrics(sync_d_loss, sync_g_loss, epoch)
                log_generated_images(generated_images)
            
            # Evaluate GAN metrics (IS and FID) periodically
            eval_interval = config["training"].get("eval_interval", 5)  # Default every 5 epochs
            if (epoch + 1) % eval_interval == 0:
                print(f"Evaluating GAN metrics at epoch {epoch + 1}...")
                try:
                    is_mean, is_std, fid_score, num_real_samples = evaluate_gan_metrics(
                        generator.module,
                        train_dataloader,
                        device,
                        config["model"]["latent_dim"],
                        num_samples=min(5000, len(train_dataloader.dataset)),  # Limit for faster evaluation
                        batch_size=32
                    )
                    
                    print(f"IS: {is_mean:.3f} ± {is_std:.3f}, FID: {fid_score:.3f}")
                    
                    # Log to wandb
                    if use_wandb:
                        log_evaluation_metrics(is_mean, is_std, fid_score, epoch)
                        
                except Exception as e:
                    print(f"Warning: Failed to evaluate GAN metrics: {e}")
            
            # Save checkpoint and plots periodically
            if (epoch + 1) % config["training"]["save_interval"] == 0:
                save_checkpoint(
                    generator.module, discriminator.module, g_optimizer, 
                    d_optimizer, epoch + 1, config["output"]["checkpoint_dir"]
                )
                plot_losses(g_losses, d_losses, 
                          os.path.join(config["output"]["log_dir"], f"losses_epoch_{epoch + 1}.png"))
    
    # Final cleanup and saving (rank 0 only)
    if rank == 0:
        print("Training completed!")
        save_final_outputs(generator, discriminator, g_optimizer, d_optimizer, 
                          g_losses, d_losses, config)
        
        if use_wandb:
            finish_wandb()
    
    cleanup_ddp()


def main():
    """Main function to setup and launch distributed training."""
    parser = argparse.ArgumentParser(description="Train GAN with DDP on multiple GPUs")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to configuration file (default: config.yaml)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No GPUs available for training")
    
    print(f"Starting distributed training on {world_size} GPUs")
    mp.spawn(train_gan_ddp, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()