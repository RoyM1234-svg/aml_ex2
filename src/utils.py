import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

def plot_trajectories(trajectories: list[list[torch.Tensor]],
                      title: str,
                      save_path: str | None = None):
    plt.figure(figsize=(10, 10))
    
    n_steps = len(trajectories[0])
    
    colors = cm.viridis(np.linspace(0, 1, n_steps))
    
    for i, trajectory in enumerate(trajectories):
        trajectory_np = np.array([t.detach().cpu().numpy() for t in trajectory])
        
        for j in range(len(trajectory_np) - 1):
            plt.plot(trajectory_np[j:j+2, 0], trajectory_np[j:j+2, 1], 
                    color=colors[j], linewidth=2, alpha=0.8)
        
        plt.scatter(trajectory_np[0, 0], trajectory_np[0, 1], 
                   color='red', s=100, marker='o', edgecolor='black',
                   label='Start' if i == 0 else "", zorder=5)
        
        plt.scatter(trajectory_np[-1, 0], trajectory_np[-1, 1], 
                   color='blue', s=100, marker='s', edgecolor='black',
                   label='End' if i == 0 else "", zorder=5)
    
    sm = cm.ScalarMappable(cmap=cm.viridis, 
                          norm=plt.Normalize(vmin=0, vmax=n_steps-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label('Step Number', rotation=270, labelpad=20)
    
    plt.title(title, fontsize=14)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
    if len(trajectories) > 0:
        plt.legend(loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_test_metrics(test_epoch_losses, test_epoch_log_probs, test_epoch_log_dets, epochs):
    plt.figure(figsize=(10, 6))
    
    epochs_range = range(1, epochs + 1)
    
    plt.plot(epochs_range, test_epoch_losses, 'b-', label='Test Loss', linewidth=2)
    plt.plot(epochs_range, test_epoch_log_probs, 'r-', label='Test Log Prob', linewidth=2)
    plt.plot(epochs_range, test_epoch_log_dets, 'g-', label='Test Log Det', linewidth=2)
    plt.title('Test Losses during training')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    
    plt.show()

def plot_samples(samples, title="Generated Samples", figsize=(10, 8), save_path=None):
    if torch.is_tensor(samples):
        points = samples.detach().cpu().numpy()
    else:
        points = samples
    
    plt.figure(figsize=figsize)
    plt.scatter(points[:, 0], points[:, 1], alpha=0.6, s=1)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def sample_points_inside_outside_rings():
    inside_points = [
        [0.1, 0.05], 
        [1.95, -0.1],
        [3.1, -0.95],
    ]
    
    outside_points = [
        [3, 1],
        [5.5, 0],
    ]
    
    all_points = np.array(inside_points + outside_points)
    mean = np.array([2, -0.5])
    std = np.array([1.5, 0.7])
    normalized = (all_points - mean) / std
    
    return torch.tensor(normalized[:3], dtype=torch.float32), torch.tensor(normalized[3:], dtype=torch.float32)

# def visualize_points_with_rings(model, inside_points, outside_points):
#     """Visualize selected points together with the Olympic rings."""
#     plt.figure(figsize=(10, 8))
    
#     # Sample from model to show the rings
#     with torch.no_grad():
#         z = torch.randn(10000, 2)
#         model_samples = model(z).numpy()
    
#     # Plot the rings
#     plt.scatter(model_samples[:, 0], model_samples[:, 1], 
#                 s=1, alpha=0.3, color='lightblue', label='Olympic rings')
    
#     # Plot inside points (in ring centers)
#     plt.scatter(inside_points[:, 0], inside_points[:, 1], 
#                 s=200, color='orange', marker='*', 
#                 label=f'Inside ring centers ({len(inside_points)})', 
#                 edgecolor='black', linewidth=2)
    
#     # Plot outside points
#     plt.scatter(outside_points[:, 0], outside_points[:, 1], 
#                 s=200, color='red', marker='X', 
#                 label=f'Outside rings ({len(outside_points)})', 
#                 edgecolor='black', linewidth=2)
    
#     # Add labels
#     for i, p in enumerate(inside_points):
#         plt.annotate(f'I{i+1}', (p[0], p[1]), xytext=(5, 5), 
#                     textcoords='offset points', fontsize=12)
#     for i, p in enumerate(outside_points):
#         plt.annotate(f'O{i+1}', (p[0], p[1]), xytext=(5, 5), 
#                     textcoords='offset points', fontsize=12)
    
#     plt.legend()
#     plt.title('Selected Points vs Olympic Rings Distribution')
#     plt.grid(True, alpha=0.3)
#     plt.axis('equal')
#     plt.xlim(-3, 3)
#     plt.ylim(-3, 3)
#     plt.show()