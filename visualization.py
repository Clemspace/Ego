import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_training_progress(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def visualize_prediction(model, input_grid):
    model.eval()
    with torch.no_grad():
        input_tensor = input_grid.unsqueeze(0).unsqueeze(0)
        prediction = model(input_tensor).squeeze().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(input_grid.numpy(), ax=ax1, cmap='viridis', cbar=False)
    ax1.set_title('Input Grid')
    sns.heatmap(prediction, ax=ax2, cmap='viridis', cbar=False)
    ax2.set_title('Predicted Output')
    plt.show()