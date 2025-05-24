import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import your models and data functions
from normalized_flow_model import NormalizedFlowModel
from normalized_flow_loss import NormalizedFlowLoss
from create_data import create_unconditional_olympic_rings 


def train_normalizing_flow():
    learning_rate = 1e-3
    num_data_points = 250_000
    epochs = 7
    batch_size = 128
    input_dim = 2
    n_layers = 15
    
    print("Generating Olympic rings training data...")
    data_np = create_unconditional_olympic_rings(num_data_points, ring_thickness=0.25, verbose=True)
    data = torch.tensor(data_np, dtype=torch.float32)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NormalizedFlowModel(input_dim, n_layers=n_layers)
    loss_fn = NormalizedFlowLoss(input_dim, model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    
    model.train()
    epoch_losses = []

    for epoch in range(epochs):
        loss_sum = 0

        for (batch_data,) in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()
            loss = loss_fn(batch_data)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        avg_loss = loss_sum / len(dataloader)
        epoch_losses.append(avg_loss)
        scheduler.step()

        
        print(f'Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}')
    
    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood')
    plt.grid(True)
    plt.show()
    
    print(f"Final loss: {epoch_losses[-1]:.4f}")
    
    return model, epoch_losses

if __name__ == "__main__":
    # Run training
    model, epoch_losses = train_normalizing_flow()

    # Save model
    torch.save(model.state_dict(), "normalized_flow_model.pth")

    # Load model
    # model.load_state_dict(torch.load("normalized_flow_model.pth"))

    # Test model