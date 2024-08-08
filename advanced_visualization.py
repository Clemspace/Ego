import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchviz import make_dot
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def visualize_attention(model, input_grid):
    model.eval()
    with torch.no_grad():
        input_tensor = input_grid.unsqueeze(0).unsqueeze(0)
        output = model(input_tensor)
        attention_weights = model.attention.get_attention_weights()
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(attention_weights.squeeze().numpy(), cmap='viridis')
    plt.title('Attention Weights')
    plt.show()

def plot_learning_rate(optimizer):
    lrs = [group['lr'] for group in optimizer.param_groups]
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()

def visualize_model_architecture(model, input_size=(1, 1, 30, 30)):
    x = torch.randn(input_size)
    y = model(x)
    dot = make_dot(y.mean(), params=dict(model.named_parameters()))
    dot.render("model_architecture", format="png")
    print("Model architecture visualization saved as 'model_architecture.png'")

def visualize_grad_cam(model, input_grid, target_layer):
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=torch.cuda.is_available())
    input_tensor = input_grid.unsqueeze(0).unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor)
    visualization = show_cam_on_image(input_grid.numpy(), grayscale_cam[0, :], use_rgb=True)
    
    plt.imshow(visualization)
    plt.title('Grad-CAM Visualization')
    plt.axis('off')
    plt.show()

def visualize_embeddings(model, dataset, method='tsne'):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for input_grid, output_grid in dataset:
            input_tensor = input_grid.unsqueeze(0).unsqueeze(0)
            embedding = model.get_embedding(input_tensor).squeeze().numpy()
            embeddings.append(embedding)
            labels.append(output_grid.max().item())  # Assuming the max value represents the class
    
    embeddings = np.array(embeddings)
    
    if method == 'tsne':
        reduced_embeddings = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    elif method == 'pca':
        reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'Embeddings visualization using {method.upper()}')
    plt.show()

def plot_parameter_distributions(model):
    plt.figure(figsize=(15, 5))
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            plt.subplot(1, 3, i+1)
            sns.histplot(param.detach().cpu().numpy().flatten(), kde=True)
            plt.title(f'{name} distribution')
    plt.tight_layout()
    plt.show()
