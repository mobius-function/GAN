import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


def save_samples(generator, fixed_noise, epoch, sample_dir, device):
    """
    Generate and save sample images
    """
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise.to(device))
        fake_images = fake_images.detach().cpu()
        
        os.makedirs(sample_dir, exist_ok=True)
        
        vutils.save_image(
            fake_images,
            os.path.join(sample_dir, f'fake_samples_epoch_{epoch}.png'),
            normalize=True,
            nrow=8
        )
    generator.train()


def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch, checkpoint_dir):
    """
    Save model checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }
    
    torch.save(
        checkpoint,
        os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    )


def load_checkpoint(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    return checkpoint['epoch']


def plot_losses(g_losses, d_losses, save_path):
    """
    Plot generator and discriminator losses
    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G Loss")
    plt.plot(d_losses, label="D Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False