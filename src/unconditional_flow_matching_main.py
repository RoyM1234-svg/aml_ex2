import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from flow_matching_models import UnconditionalFlowMatchingModel
from create_data import create_unconditional_olympic_rings
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from unconditional_flow_matching_loss import UnconditionalFlowMatchingLoss
from utils import plot_samples, plot_trajectories, sample_points_inside_outside_rings
import matplotlib.pyplot as plt
import os


def plot_training_losses(batch_losses, save_path="training_losses.png"):
    """Plot the training losses across batches."""
    plt.figure(figsize=(10, 6))
    plt.plot(batch_losses)
    plt.title('Training Loss vs Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training loss plot saved to {save_path}")

def train_unconditional_flow_matching_model():   
    learning_rate = 1e-3
    num_data_points = 270000
    epochs = 20
    batch_size = 128
    hidden_dim = 64

    model = UnconditionalFlowMatchingModel(input_dim=2, hidden_dim=hidden_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    data = create_unconditional_olympic_rings(num_data_points)
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train_data = torch.tensor(train, dtype=torch.float32)
    test_data = torch.tensor(test, dtype=torch.float32)
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = UnconditionalFlowMatchingLoss(model, input_dim=2)
    batch_losses = []

    for epoch in range(epochs):
        model.train()
        for (batch_data,) in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            loss = loss_fn(batch_data)
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        scheduler.step()

    return model, batch_losses

def sample_from__uncoditional_model(model, delta_t, num_samples=2000, times_to_plot=[1]):
    os.makedirs("uncoditional_flow_matching_samples", exist_ok=True)
    model.eval()
    base_dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    y = base_dist.sample((num_samples,))
    with torch.no_grad():
        for t in torch.arange(0, 1 + delta_t, delta_t):
            if t in times_to_plot:
                plot_samples(y, title=f"Flow Matching Generated Samples at t={t} with delta_t={delta_t}", 
                             save_path=f"uncoditional_flow_matching_samples/t={t}_delta_t={delta_t}.png")
            t_tensor = torch.full((y.shape[0], 1), t.item())
            y += model(y, t_tensor) * delta_t
    return y

def sample_trajectories(model, num_samples, delta_t):
    model.eval()
    base_dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    y = base_dist.sample((num_samples,))
    trajectories = create_trajectories_for_points(model, y, delta_t)
    plot_trajectories(trajectories, 
                      title="Trajectories of Points Through Unconditional Flow Matching",
                      save_path=f"unconditional_flow_matching_trajectories.png")
    
def create_trajectories_for_points(model: UnconditionalFlowMatchingModel,
                                   points: torch.Tensor, 
                                   delta_t: float) -> list[list[torch.Tensor]]:
    model.eval()
    current = points.clone()
    trajectories = [[point.clone()] for point in current]
    with torch.no_grad():
        for t in torch.arange(0, 1 + delta_t, delta_t):
            t_tensor = torch.full((current.shape[0], 1), t.item())
            current += model(current, t_tensor) * delta_t
            for i, point in enumerate(current):
                trajectories[i].append(point.clone())
    return trajectories
    
def create_reverse_trajectories_for_points(model, points, delta_t) -> list[list[torch.Tensor]]:
    model.eval()
    current = points.clone()
    trajectories = [[point.clone()] for point in current]
    with torch.no_grad():
        for t in torch.arange(1, 0 - delta_t, -delta_t):
            t_tensor = torch.full((current.shape[0], 1), t.item())
            current += model(current, t_tensor) * delta_t
            for i, point in enumerate(current):
                trajectories[i].append(point.clone())
    return trajectories

def compare_reversed_points(model, delta_t):
    inside_points, outside_points = sample_points_inside_outside_rings()
    
    original_points = torch.cat([inside_points, outside_points])
    print("original_points", original_points)

    trajectories = create_reverse_trajectories_for_points(model, original_points, delta_t=delta_t)

    plot_trajectories(trajectories, title="Reverse Trajectories of Points Through Unconditional Flow Matching")

    inverted_points = torch.stack([trajectory[-1] for trajectory in trajectories])

    recreated_trajectoires = create_trajectories_for_points(model, inverted_points, delta_t=delta_t)

    recreated_points = torch.stack([trajectory[-1] for trajectory in recreated_trajectoires])

    print("recreated_points", recreated_points)



    

def main():
    # Train model
    # model, batch_losses = train_unconditional_flow_matching_model()
    # plot_training_losses(batch_losses)
    # torch.save(model.state_dict(), "unconditional_flow_matching_model.pth")

    #load model
    model = UnconditionalFlowMatchingModel(input_dim=2, hidden_dim=64)
    model.load_state_dict(torch.load("unconditional_flow_matching_model.pth"))

    # All questions 

    delta_t = 1/1000
    times_to_plot = [0, 0.2, 0.4, 0.6, 0.8, 1]
    sample_from__uncoditional_model(model, delta_t, times_to_plot=times_to_plot)

    sample_trajectories(model, 10, delta_t)

    delta_ts = [0.002, 0.02, 0.05, 0.1, 0.2]
    for delta_t in delta_ts:
        sample_from__uncoditional_model(model, delta_t, times_to_plot=[1])

    compare_reversed_points(model, delta_t)
    


    

if __name__ == "__main__":
    main()