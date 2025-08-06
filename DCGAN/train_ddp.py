import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm

from models import Generator, Discriminator
from dataset_celebrity import get_dataloader, CelebrityFacesDataset
from utils import save_samples, save_checkpoint, plot_losses, set_seed


def setup(rank, world_size):
    """Initialize the distributed environment"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_dataloader_ddp(config, rank, world_size):
    """Get dataloader with DistributedSampler for DDP"""
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose(
        [
            transforms.Resize(config["dataset"]["image_size"]),
            transforms.CenterCrop(config["dataset"]["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if config["dataset"]["name"] == "celebrity":
        # Use celebrity dataset from Hugging Face
        max_images = config["dataset"].get("max_images", None)
        dataset = CelebrityFacesDataset(transform=transform, max_images=max_images)
    else:
        from torchvision.datasets import CelebA

        dataset = CelebA(
            root=config["dataset"]["data_dir"],
            split="train",
            transform=transform,
            download=False,  # Assume already downloaded
        )

    # Create distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # Adjust batch size per GPU
    batch_size_per_gpu = config["dataset"]["batch_size"] // world_size

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    return dataloader, sampler


def train_gan_ddp(rank, world_size, config):
    """Main training function for DDP"""
    setup(rank, world_size)

    # Set seed for reproducibility
    set_seed(config["seed"] + rank)

    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Only rank 0 handles logging
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(config["output"]["log_dir"], "tensorboard"))
        print(f"Using {world_size} GPUs for training")

    # Get dataloader with distributed sampler
    dataloader, sampler = get_dataloader_ddp(config, rank, world_size)

    if rank == 0:
        print(f"Loaded dataset with {len(dataloader)} batches per GPU")

    # Create models
    generator = Generator(
        latent_dim=config["model"]["latent_dim"],
        channels=config["model"]["generator"]["channels"],
        output_size=config["dataset"]["image_size"],
    ).to(device)

    discriminator = Discriminator(
        channels=config["model"]["discriminator"]["channels"], input_size=config["dataset"]["image_size"]
    ).to(device)

    # Wrap models with DDP
    generator = DDP(generator, device_ids=[rank])
    discriminator = DDP(discriminator, device_ids=[rank])

    # Optimizers
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

    criterion = nn.BCELoss()

    # Fixed noise for consistent validation image generation (8 images)
    fixed_noise = torch.randn(8, config["model"]["latent_dim"]).to(device)

    # Create output directories (only rank 0)
    if rank == 0:
        os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
        os.makedirs(config["output"]["sample_dir"], exist_ok=True)
        os.makedirs(config["output"]["log_dir"], exist_ok=True)

    g_losses = []
    d_losses = []

    if rank == 0:
        print("Starting training...")

    for epoch in range(config["training"]["num_epochs"]):
        # Set epoch for distributed sampler (important for shuffling)
        sampler.set_epoch(epoch)

        epoch_g_loss = 0
        epoch_d_loss = 0

        # Progress bar only on rank 0
        if rank == 0:
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        else:
            progress_bar = dataloader

        for i, (real_images, _) in enumerate(progress_bar):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            d_optimizer.zero_grad()

            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            z = torch.randn(batch_size, config["model"]["latent_dim"]).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            z = torch.randn(batch_size, config["model"]["latent_dim"]).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)

            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

            # Logging (only rank 0)
            if rank == 0:
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

                if (i + 1) % config["training"]["log_interval"] == 0:
                    progress_bar.set_postfix({"D Loss": f"{d_loss.item():.4f}", "G Loss": f"{g_loss.item():.4f}"})

                    # Log step-level metrics to TensorBoard
                    global_step = epoch * len(dataloader) + i
                    writer.add_scalar("Loss/Discriminator", d_loss.item(), global_step)
                    writer.add_scalar("Loss/Generator", g_loss.item(), global_step)

                if (epoch * len(dataloader) + i) % config["training"]["sample_interval"] == 0:
                    save_samples(generator.module, fixed_noise, epoch, config["output"]["sample_dir"], device)

        # Synchronize metrics across GPUs
        avg_g_loss_tensor = torch.tensor(epoch_g_loss / len(dataloader)).to(device)
        avg_d_loss_tensor = torch.tensor(epoch_d_loss / len(dataloader)).to(device)
        dist.all_reduce(avg_g_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_d_loss_tensor, op=dist.ReduceOp.SUM)
        avg_g_loss = avg_g_loss_tensor.item() / world_size
        avg_d_loss = avg_d_loss_tensor.item() / world_size

        if rank == 0:
            print(
                f"Epoch [{epoch + 1}/{config['training']['num_epochs']}] "
                f"D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}"
            )

            # Generate validation images with fixed noise
            generator.eval()
            with torch.no_grad():
                validation_images = generator(fixed_noise)
                validation_images = validation_images.detach().cpu()
            generator.train()

            # Log epoch-level metrics to TensorBoard
            writer.add_scalar("Epoch_Loss/Discriminator", avg_d_loss, epoch + 1)
            writer.add_scalar("Epoch_Loss/Generator", avg_g_loss, epoch + 1)

            # Log validation images to TensorBoard (8 fixed images in a grid)
            img_grid = vutils.make_grid(validation_images, nrow=4, normalize=True)
            writer.add_image("Validation_Images", img_grid, epoch + 1)

            if (epoch + 1) % config["training"]["save_interval"] == 0:
                save_checkpoint(
                    generator.module,
                    discriminator.module,
                    g_optimizer,
                    d_optimizer,
                    epoch + 1,
                    config["output"]["checkpoint_dir"],
                )
                plot_losses(
                    g_losses, d_losses, os.path.join(config["output"]["log_dir"], f"losses_epoch_{epoch + 1}.png")
                )

    if rank == 0:
        print("Training completed!")
        save_checkpoint(
            generator.module,
            discriminator.module,
            g_optimizer,
            d_optimizer,
            config["training"]["num_epochs"],
            config["output"]["checkpoint_dir"],
        )
        plot_losses(g_losses, d_losses, os.path.join(config["output"]["log_dir"], "final_losses.png"))
        writer.close()

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Train GAN with DDP on multiple GPUs")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Get number of GPUs
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No GPUs available for training")

    # Spawn processes for DDP
    mp.spawn(train_gan_ddp, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
