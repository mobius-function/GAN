"""Distributed Data Parallel utilities for DCGAN training."""

import os
import torch
import torch.distributed as dist


def setup_ddp(rank: int, world_size: int, master_port: str = "12355"):
    """Initialize the distributed environment.
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        master_port: Port for master process communication
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def setup_device(rank: int) -> torch.device:
    """Setup device for the current process.
    
    Args:
        rank: Current process rank
        
    Returns:
        torch.device: CUDA device for this process
    """
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def synchronize_metrics(metrics_dict: dict, world_size: int, device: torch.device) -> dict:
    """Synchronize metrics across all GPUs.
    
    Args:
        metrics_dict: Dictionary of metric names to values
        world_size: Total number of processes
        device: Current device
        
    Returns:
        dict: Dictionary with averaged metrics
    """
    synchronized_metrics = {}
    
    for name, value in metrics_dict.items():
        # Convert to tensor and move to device
        tensor_value = torch.tensor(value, dtype=torch.float32).to(device)
        
        # All-reduce to sum across all processes
        dist.all_reduce(tensor_value, op=dist.ReduceOp.SUM)
        
        # Average by world size
        synchronized_metrics[name] = tensor_value.item() / world_size
    
    return synchronized_metrics