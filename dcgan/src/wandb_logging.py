"""WandB logging utilities for DCGAN training."""

import torchvision.utils as vutils
import wandb


def log_step_metrics(d_loss, g_loss, epoch, step, steps_per_epoch):
    """Log step-level metrics to wandb.
    
    Args:
        d_loss: Discriminator loss tensor
        g_loss: Generator loss tensor
        epoch: Current epoch number
        step: Current step within epoch
        steps_per_epoch: Total steps per epoch
    """
    total_steps = epoch * steps_per_epoch + step
    wandb.log({
        "Loss/Discriminator": d_loss.item(),
        "Loss/Generator": g_loss.item(),
        "step": total_steps
    })


def log_epoch_metrics(avg_d_loss, avg_g_loss, epoch):
    """Log epoch-level metrics to wandb.
    
    Args:
        avg_d_loss: Average discriminator loss for the epoch
        avg_g_loss: Average generator loss for the epoch
        epoch: Current epoch number
    """
    wandb.log({
        "Loss/Discriminator": avg_d_loss,
        "Loss/Generator": avg_g_loss,
        "epoch": epoch + 1
    })


def log_generated_images(generated_images, nrow=4):
    """Log generated images to wandb.
    
    Args:
        generated_images: Tensor of generated images
        nrow: Number of images per row in the grid
    """
    img_grid = vutils.make_grid(generated_images, nrow=nrow, normalize=True)
    wandb.log({"Generated_Images": wandb.Image(img_grid)})


def setup_wandb(config, rank):
    """Setup wandb logging for the specified rank.
    
    Args:
        config: Configuration dictionary containing wandb settings
        rank: Process rank (only rank 0 should log)
        
    Returns:
        bool: True if wandb is enabled and setup successful, False otherwise
    """
    if rank != 0:
        return False
    
    try:
        wandb_config = config.get("wandb", {})
        wandb_enabled = wandb_config.get("enabled", False)
        
        if not wandb_enabled:
            print("WandB logging is disabled in config")
            return False
        
        wandb_mode = wandb_config.get("mode", "offline")
        wandb_project = wandb_config.get("project", "dcgan")
        wandb_run_name = wandb_config.get("run_name") or f"dcgan-ddp-{config['dataset']['name']}"
        wandb_api_key = wandb_config.get("api_key")
        
        # Set API key if provided in config
        if wandb_api_key:
            import os
            os.environ['WANDB_API_KEY'] = wandb_api_key
        
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config,
            mode=wandb_mode
        )
        
        print(f"WandB project: {wandb_project} (mode: {wandb_mode})")
        return True
        
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")
        print("Continuing without wandb logging...")
        return False


def log_evaluation_metrics(is_mean: float, is_std: float, fid_score: float, epoch: int):
    """Log evaluation metrics (IS and FID) to wandb.
    
    Args:
        is_mean: Mean Inception Score
        is_std: Standard deviation of Inception Score  
        fid_score: Fréchet Inception Distance score
        epoch: Current epoch number
    """
    wandb.log({
        "Metrics/Inception_Score": is_mean,
        "Metrics/Inception_Score_Std": is_std,
        "Metrics/FID_Score": fid_score,
        "epoch": epoch + 1
    })


def finish_wandb():
    """Finish wandb run."""
    wandb.finish()