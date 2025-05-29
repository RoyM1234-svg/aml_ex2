import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_trajectories(trajectories):
    plt.figure(figsize=(10, 10))
    
    n_steps = len(trajectories[0])
    
    colors = cm.viridis(np.linspace(0, 1, n_steps))
    
    for i, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)  
        
        for j in range(len(trajectory) - 1):
            plt.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1], 
                    color=colors[j], linewidth=2, alpha=0.8)
        
        plt.scatter(trajectory[0, 0], trajectory[0, 1], 
                   color='red', s=100, marker='o', edgecolor='black',
                   label='Start' if i == 0 else "", zorder=5)
        
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                   color='blue', s=100, marker='s', edgecolor='black',
                   label='End' if i == 0 else "", zorder=5)
    
    sm = cm.ScalarMappable(cmap=cm.viridis, 
                          norm=plt.Normalize(vmin=0, vmax=n_steps-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar.set_label('Step Number', rotation=270, labelpad=20)
    
    plt.title(f'Trajectories of {len(trajectories)} Points Through Normalizing Flow', fontsize=14)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    
    if len(trajectories) > 0:
        plt.legend(loc='upper right')
    
    plt.tight_layout()
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