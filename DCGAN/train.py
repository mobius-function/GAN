import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import Generator, Discriminator
from dataset import get_celeba_dataloader
from utils import save_samples, save_checkpoint, plot_losses, set_seed


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_gan(config):
    """Main training function"""
    
    set_seed(config['seed'])
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataloader = get_celeba_dataloader(config)
    print(f"Loaded dataset with {len(dataloader)} batches")
    
    generator = Generator(
        latent_dim=config['model']['latent_dim'],
        channels=config['model']['generator']['channels'],
        output_size=config['dataset']['image_size']
    ).to(device)
    
    discriminator = Discriminator(
        channels=config['model']['discriminator']['channels'],
        input_size=config['dataset']['image_size']
    ).to(device)
    
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=config['training']['learning_rate']['generator'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config['training']['learning_rate']['discriminator'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    
    criterion = nn.BCELoss()
    
    fixed_noise = torch.randn(64, config['model']['latent_dim'])
    
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output']['sample_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    
    g_losses = []
    d_losses = []
    
    print("Starting training...")
    
    for epoch in range(config['training']['num_epochs']):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for i, (real_images, _) in enumerate(progress_bar):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            d_optimizer.zero_grad()
            
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            
            z = torch.randn(batch_size, config['model']['latent_dim']).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, config['model']['latent_dim']).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            
            g_loss.backward()
            g_optimizer.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            if (i + 1) % config['training']['log_interval'] == 0:
                progress_bar.set_postfix({
                    'D Loss': f"{d_loss.item():.4f}",
                    'G Loss': f"{g_loss.item():.4f}"
                })
            
            if (epoch * len(dataloader) + i) % config['training']['sample_interval'] == 0:
                save_samples(generator, fixed_noise, epoch, config['output']['sample_dir'], device)
        
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}] "
              f"D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
        
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_checkpoint(
                generator, discriminator, g_optimizer, d_optimizer,
                epoch + 1, config['output']['checkpoint_dir']
            )
            plot_losses(
                g_losses, d_losses,
                os.path.join(config['output']['log_dir'], f'losses_epoch_{epoch+1}.png')
            )
    
    print("Training completed!")
    save_checkpoint(
        generator, discriminator, g_optimizer, d_optimizer,
        config['training']['num_epochs'], config['output']['checkpoint_dir']
    )
    plot_losses(
        g_losses, d_losses,
        os.path.join(config['output']['log_dir'], 'final_losses.png')
    )


def main():
    parser = argparse.ArgumentParser(description='Train GAN on CelebA dataset')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_gan(config)


if __name__ == '__main__':
    main()