import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

def load_wandb_data(run_id):
    """Load wandb data from local run file."""
    wandb_file = f"./wandb/run-*-{run_id}/run-{run_id}.wandb"
    
    # Find the actual file path
    import glob
    files = glob.glob(wandb_file)
    if not files:
        print(f"No wandb file found for run {run_id}")
        return None
    
    wandb_file_path = files[0]
    print(f"Loading data from: {wandb_file_path}")
    
    # Load the wandb file (it's a binary format, so we'll try to read the summary)
    summary_file = wandb_file_path.replace(f"run-{run_id}.wandb", "wandb-summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            print(f"Summary keys for {run_id}: {list(summary.keys())}")
    
    # Try to find metrics from the config and summary
    config_file = wandb_file_path.replace(f"run-{run_id}.wandb", "config.yaml")
    if os.path.exists(config_file):
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Config loaded for {run_id}")
    
    return None

def extract_losses_from_logs(run_id):
    """Extract losses from output logs since wandb files are binary."""
    log_file = f"./wandb/run-*-{run_id}/files/output.log"
    
    import glob
    files = glob.glob(log_file)
    if not files:
        print(f"No log file found for run {run_id}")
        return None, None, None
    
    log_file_path = files[0]
    print(f"Reading logs from: {log_file_path}")
    
    d_losses = []
    g_losses = []
    epochs = []
    learning_rates = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Look for loss patterns in logs
            # Format: Epoch [1/100] D Loss: 1.2551, G Loss: 0.6931
            if "D Loss:" in line and "G Loss:" in line and "Epoch [" in line:
                try:
                    parts = line.split()
                    
                    # Find epoch number
                    epoch_idx = parts.index("Epoch")
                    epoch_part = parts[epoch_idx + 1]  # [1/100]
                    epoch = int(epoch_part.split('[')[1].split('/')[0])
                    
                    # Find losses
                    d_loss_idx = parts.index("Loss:") + 1  # First occurrence is D Loss
                    d_loss = float(parts[d_loss_idx].rstrip(','))
                    
                    g_loss_idx = len(parts) - 1  # Last part is G Loss value
                    g_loss = float(parts[g_loss_idx])
                    
                    d_losses.append(d_loss)
                    g_losses.append(g_loss)
                    epochs.append(epoch)
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line.strip()}")
                    continue
            
            # Look for learning rate patterns
            if "lr:" in line.lower() or "learning_rate" in line.lower():
                try:
                    # Extract learning rate if found
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "lr:" in part.lower() or "learning_rate" in part.lower():
                            if i + 1 < len(parts):
                                lr = float(parts[i + 1].rstrip(','))
                                learning_rates.append(lr)
                                break
                except (ValueError, IndexError):
                    continue
    
    print(f"Run {run_id}: Found {len(d_losses)} loss entries, {len(learning_rates)} LR entries")
    return d_losses, g_losses, epochs

# Define the runs to plot
run_info = {
    '5a19ime1': {'label': 'Constant learning rate 0.0002', 'color': 'blue'},
    'ohpwfqva': {'label': 'Cosine annealing LR (0.0002, 0.00001)', 'color': 'red'},
    'mrh5evxb': {'label': 'Label smoothing + cosine annealing LR (0.0002, 0.00001)', 'color': 'green'}
}

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Training Run Results (Anime Dataset - images: 24000, batch size: 800, epochs: 100)', fontsize=16)

# Load data for each run
all_data = {}
for run_id in run_info.keys():
    print(f"\nProcessing run {run_id}...")
    d_losses, g_losses, epochs = extract_losses_from_logs(run_id)
    all_data[run_id] = {
        'd_losses': d_losses,
        'g_losses': g_losses,
        'epochs': epochs
    }

# Plot discriminator losses
for run_id, info in run_info.items():
    data = all_data[run_id]
    if data['d_losses']:
        axes[0].plot(data['epochs'], data['d_losses'], 
                    label=info['label'], color=info['color'], alpha=0.7)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Discriminator Loss')
axes[0].set_title('Discriminator Loss Comparison')
axes[0].grid(True, alpha=0.3)

# Plot generator losses
for run_id, info in run_info.items():
    data = all_data[run_id]
    if data['g_losses']:
        axes[1].plot(data['epochs'], data['g_losses'], 
                    label=info['label'], color=info['color'], alpha=0.7)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Generator Loss')
axes[1].set_title('Generator Loss Comparison')
axes[1].grid(True, alpha=0.3)

# Plot learning rate schedules
# For 5a19ime1: constant LR of 0.0002
# For others: we'll need to calculate cosine annealing
import numpy as np

epochs_range = np.arange(1, 101)  # 100 epochs

# Run 5a19ime1: Fixed LR
axes[2].axhline(y=0.0002, color=run_info['5a19ime1']['color'], 
               label=run_info['5a19ime1']['label'], alpha=0.7)

# Cosine annealing for other runs (T_max=100, eta_min=0.00001, initial_lr=0.0002)
T_max = 100
eta_min = 0.00001
initial_lr = 0.0002

cosine_lr = eta_min + (initial_lr - eta_min) * 0.5 * (1 + np.cos(np.pi * epochs_range / T_max))

axes[2].plot(epochs_range, cosine_lr, color=run_info['ohpwfqva']['color'],
            label=run_info['ohpwfqva']['label'], alpha=0.7)
axes[2].plot(epochs_range, cosine_lr, color=run_info['mrh5evxb']['color'],
            label=run_info['mrh5evxb']['label'], alpha=0.7, linestyle='--')

axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title('Learning Rate Schedule')
axes[2].grid(True, alpha=0.3)
axes[2].set_yscale('log')

# Create a single legend below all subplots
handles, labels = axes[1].get_legend_handles_labels()  # Get from middle plot
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for legend
plt.savefig('wandb_runs_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved as 'wandb_runs_comparison.png'")
print("Summary:")
for run_id, data in all_data.items():
    print(f"  {run_id}: {len(data['d_losses'])} data points")