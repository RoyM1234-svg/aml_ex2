import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from normalized_flow_model import NormalizedFlowModel
from normalized_flow_loss import NormalizedFlowLoss
from create_data import create_unconditional_olympic_rings 
from sklearn.model_selection import train_test_split

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

if __name__ == "__main__":

    # Train model
    model = train_normalizing_flow()

    torch.save(model.state_dict(), "normalized_flow_model.pth")

    # Load model
    input_dim = 2
    
    model = NormalizedFlowModel(input_dim)
    model.load_state_dict(torch.load("normalized_flow_model.pth"))
    

    sample_from_model(model, num_samples=10000)