from flow_matching_models import ConditionalFlowMatchingModel
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn.functional as F
from tqdm import tqdm
from create_data import create_olympic_rings
from conditional_flow_matching_loss import ConditionalFlowMatchingLoss
import matplotlib.pyplot as plt

def train_conditional_flow_model():
    learning_rate = 1e-3
    num_data_points = 270000
    epochs = 20
    batch_size = 128
    hidden_dim = 64

    model = ConditionalFlowMatchingModel(input_dim=2, hidden_dim=hidden_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train, labels, int_to_label = create_olympic_rings(num_data_points)
    labels = torch.tensor(labels)
    train = torch.tensor(train, dtype=torch.float32)
    train_dataset = TensorDataset(train, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = ConditionalFlowMatchingLoss(model, input_dim=2)

    for epoch in range(epochs):
        for (batch_data,labels) in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            loss = loss_fn(batch_data, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()    
    
    return model    

def sample_from_conditional_flow_model(model: ConditionalFlowMatchingModel, num_samples, num_classes, delta_t):
    model.eval()
    labels = torch.randint(0, num_classes, (num_samples,))
    samples = torch.randn(num_samples, 2)

    for t in torch.arange(0, 1+delta_t, delta_t):
        t_tensor = torch.full((num_samples, 1), t.item())
        samples = samples + delta_t * model(samples, t_tensor, labels)

    return samples, labels

def sample_from_each_class(model: ConditionalFlowMatchingModel, num_classes: int, delta_t):
    model.eval()
    labels = torch.arange(num_classes)
    samples = torch.randn(num_classes, 2)

    # Store trajectories - list of tensors, each tensor is (num_classes, 2)
    trajectories = [samples.clone()]
    
    for t in torch.arange(0, 1+delta_t, delta_t):
        t_tensor = torch.full((num_classes, 1), t.item())
        samples = samples + delta_t * model(samples, t_tensor, labels)
        trajectories.append(samples.clone())

    return samples, labels, trajectories

def plot_conditional_samples(samples, labels, num_classes, title):
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    colors = ['black', 'blue', 'green', 'red', 'yellow']
    
    plt.figure(figsize=(10, 8))
    
    for class_idx in range(num_classes):
        mask = labels == class_idx
        class_samples = samples[mask]
        
        plt.scatter(class_samples[:, 0], class_samples[:, 1], 
                   c=colors[class_idx], s=10, alpha=0.6, 
                   label=f'{colors[class_idx].capitalize()} Ring')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_class_trajectories(trajectories, labels, num_classes, title="Flow Trajectories"):
    """Plot the trajectories showing how samples evolve from noise to final samples"""
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    colors = ['black', 'blue', 'green', 'red', 'yellow']
    
    plt.figure(figsize=(12, 10))
    
    for class_idx in range(num_classes):
        # Extract trajectory for this class
        class_trajectory = torch.stack([traj[class_idx] for traj in trajectories])
        class_trajectory = class_trajectory.detach().cpu().numpy()
        
        # Plot the trajectory as a line
        plt.plot(class_trajectory[:, 0], class_trajectory[:, 1], 
                c=colors[class_idx], alpha=0.7, linewidth=2,
                label=f'{colors[class_idx].capitalize()} Ring Trajectory')
        
        # Mark the starting point (noise) with square marker
        plt.scatter(class_trajectory[0, 0], class_trajectory[0, 1], 
                   c=colors[class_idx], s=100, marker='s', alpha=0.8, edgecolors='white')
        
        # Mark the ending point (final sample) with circle marker
        plt.scatter(class_trajectory[-1, 0], class_trajectory[-1, 1], 
                   c=colors[class_idx], s=100, marker='o', alpha=1.0, edgecolors='white')
    
    # Add legend entries for markers
    plt.scatter([], [], c='gray', s=100, marker='s', alpha=0.8, edgecolors='white', label='Start')
    plt.scatter([], [], c='gray', s=100, marker='o', alpha=1.0, edgecolors='white', label='End')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    # model = train_conditional_flow_model()
    # torch.save(model.state_dict(), "conditional_flow_matching_model.pth")

    # load model
    delta_t = 1/1000
    model = ConditionalFlowMatchingModel(input_dim=2, hidden_dim=64)
    model.load_state_dict(torch.load("conditional_flow_matching_model.pth"))

    # Sample many samples from all classes
    # samples, labels = sample_from_conditional_flow_model(model, num_samples=3000, num_classes=5, delta_t=delta_t)
    # plot_conditional_samples(samples, labels, num_classes=5, title="Conditional Flow Matching Samples")

    # Sample one from each class and visualize trajectories
    class_samples, class_labels, trajectories = sample_from_each_class(model, num_classes=5, delta_t=delta_t)
    plot_class_trajectories(trajectories, class_labels, num_classes=5, title="Classes Trajectories")
    
    # Also plot the final samples from each class
    plot_conditional_samples(class_samples, class_labels, num_classes=5, title="One Sample from Each Class")

if __name__ == "__main__":
    main()