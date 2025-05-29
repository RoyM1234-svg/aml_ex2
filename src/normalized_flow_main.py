import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from normalized_flow_model import NormalizedFlowModel, AffineCouplingLayer, PermutationLayer
from normalized_flow_loss import NormalizedFlowLoss
from create_data import create_unconditional_olympic_rings 
from sklearn.model_selection import train_test_split
from visualizations_utils import plot_test_metrics, plot_trajectories

def sample_from_model(model, num_samples=10000):
    model.eval()
    z = torch.randn(num_samples, 2)
    
    with torch.no_grad():
        samples = model(z).numpy()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
    plt.title('Samples from Normalizing Flow')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
    return samples

def evaluate_model(model, test_dataloader, loss_fn):
    model.eval()
    loss_sum = 0
    log_prob_sum = 0
    log_det_sum = 0
    with torch.no_grad():
        for (batch_data,) in tqdm(test_dataloader):
            loss, log_prob, log_det = loss_fn(batch_data)
            loss_sum += loss.item()
            log_prob_sum += log_prob.item()
            log_det_sum += log_det.item()

    return loss_sum / len(test_dataloader), log_prob_sum / len(test_dataloader), log_det_sum / len(test_dataloader)

def train_normalizing_flow():
    learning_rate = 1e-3
    num_data_points = 270000
    epochs = 20
    batch_size = 128
    input_dim = 2
    n_layers = 15
    
    print("Generating Olympic rings training data...")
    data_np = create_unconditional_olympic_rings(num_data_points)
    train, test = train_test_split(data_np, test_size=0.2, random_state=42)

    train_data = torch.tensor(train, dtype=torch.float32)
    test_data = torch.tensor(test, dtype=torch.float32)
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = NormalizedFlowModel(input_dim, n_layers=n_layers)
    loss_fn = NormalizedFlowLoss(input_dim, model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_epoch_losses = []

    test_epoch_losses = []
    test_epoch_log_probs = []
    test_epoch_log_dets = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for (batch_data,) in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            loss, _, _= loss_fn(batch_data)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        avg_loss = loss_sum / len(train_dataloader)
        train_epoch_losses.append(avg_loss)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{epochs} - Training loss: {avg_loss:.4f}')

        test_loss, test_log_prob, test_log_det = evaluate_model(model, test_dataloader, loss_fn)
        test_epoch_losses.append(test_loss)
        test_epoch_log_probs.append(test_log_prob)
        test_epoch_log_dets.append(test_log_det)
        print(f'Epoch {epoch+1}/{epochs} - Test loss: {test_loss:.4f}')

        # sample_from_model(model, num_samples=2000)
    
    plot_test_metrics(test_epoch_losses, test_epoch_log_probs, test_epoch_log_dets, epochs)
    
    return model

def sample_through_layers(model, num_samples, save_dir="visualizations_through_layers"):
    model.eval()
    z = torch.randn(num_samples, 2)

    current = z.clone()

    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        plt.figure(figsize=(6, 6))
        plt.scatter(current.numpy()[:, 0], current.numpy()[:, 1], s=1, alpha=0.5)
        plt.title('Initial Distribution')
        plt.gca().set_aspect('equal')
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'layer_00_initial.png'), dpi=150, bbox_inches='tight')
        plt.close()

        for i, layer in enumerate(model.layers):
            current = layer(current)
            layer_type = 'AffineCoupling' if isinstance(layer, AffineCouplingLayer) else 'Permutation'
            
            plt.figure(figsize=(6, 6))
            plt.scatter(current.numpy()[:, 0], current.numpy()[:, 1], s=1, alpha=0.5)
            plt.title(f'After Layer {i+1} ({layer_type})')
            plt.gca().set_aspect('equal')
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
            plt.grid(True, alpha=0.3)
            
            filename = f'layer_{i+1:02d}_{layer_type}.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()

def sample_trajectories(model, num_samples):
    z = torch.randn(num_samples, 2)
    trajectories = create_trajectories_for_points(model, z)
    plot_trajectories(trajectories)

def create_trajectories_for_points(model, points):
    model.eval()
    current = points.clone()
    trajectories = [[] for _ in range(len(points))]
    with torch.no_grad():
        for i in range(len(points)):
            trajectories[i].append(current[i].numpy())

        for layer in model.layers:
            current = layer(current)
            for i in range(len(points)):
                trajectories[i].append(current[i].numpy())

    return trajectories

def create_reverse_trajectories_for_points(model, points):
    model.eval()
    current = points.clone()
    trajectories = [[] for _ in range(len(points))]
    with torch.no_grad():
        for i in range(len(points)):
            trajectories[i].append(current[i].numpy())

        for layer in reversed(model.layers):
            current = layer.inverse(current)
            for i in range(len(points)):
                trajectories[i].append(current[i].numpy())

    return trajectories

def sample_points_inside_outside_rings():
    centers = [(0, 0), (2, 0), (4, 0), (1, -1), (3, -1)]
    
    inside_points = []
    for i in range(3):
        center = centers[i % 5]
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, 0.3)
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        inside_points.append([x, y])
    
    outside_points = [
        [3, 1],
        [5.5, 0],
    ]
    
    all_points = np.array(inside_points + outside_points)
    mean = np.array([2, -0.5])
    std = np.array([1.5, 0.7])
    normalized = (all_points - mean) / std
    
    return torch.tensor(normalized[:3], dtype=torch.float32), torch.tensor(normalized[3:], dtype=torch.float32)




if __name__ == "__main__":

    # Train model
    # model = train_normalizing_flow()

    # torch.save(model.state_dict(), "normalized_flow_model.pth")

    # Load model
    input_dim = 2
    
    model = NormalizedFlowModel(input_dim)
    model.load_state_dict(torch.load("normalized_flow_model.pth"))
    

    sample_from_model(model, num_samples=1000)

    inside_points, outside_points = sample_points_inside_outside_rings()
    all_points = torch.cat([inside_points, outside_points])

    trajectories = create_reverse_trajectories_for_points(model, all_points)
    plot_trajectories(trajectories)
    

    print("inside points log prob", model.log_prob(inside_points))
    print("outside points log prob", model.log_prob(outside_points))
    
