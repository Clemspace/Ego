import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
from tqdm import tqdm
import os
import json
from data_utils import ARCDataset, ARCTask
from utils import get_device, print_cuda_info
import torch.nn.functional as F

# Import the new model
from modules import  MetaLearner, ModelStatistics, AdaptiveNetworkWithCorrelation



def pad_tensor(tensor, target_shape):
    """
    Pad the input tensor to match the target shape.
    The tensor is padded with zeros (or any other value if necessary).
    """
    pad_height = target_shape[0] - tensor.shape[0]
    pad_width = target_shape[1] - tensor.shape[1]
    return F.pad(tensor, (0, pad_width, 0, pad_height))

def load_arc_data(challenge_file, solution_file=None):
    with open(challenge_file, 'r') as f:
        challenges = json.load(f)
    
    if solution_file:
        with open(solution_file, 'r') as f:
            solutions = json.load(f)
    else:
        solutions = {}
    
    tasks = {}
    for task_id, task_data in challenges.items():
        if task_id in solutions:
            for test_item, solution in zip(task_data['test'], solutions[task_id]):
                test_item['output'] = solution
        tasks[task_id] = ARCTask(task_data)
    
    return tasks

def train_and_evaluate(model, train_tasks, eval_tasks, epochs):
    device = get_device()
    model = model.to(device)
    meta_learner = MetaLearner(model)
    
    first_task = next(iter(train_tasks.values()))
    first_input, _ = first_task.train[0]
    stats = ModelStatistics(model, first_input.shape)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_eval_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        total_train_samples = 0
        
        # Regular training loop
        for task in tqdm(train_tasks.values(), desc=f"Epoch {epoch+1}/{epochs} - Training"):
            for input_grid, output_grid in task.train:
                input_grid = input_grid.to(device).float()
                output_grid = output_grid.to(device)
                
                optimizer.zero_grad()
                output = model(input_grid.unsqueeze(0))  # Add batch dimension
                
                # For loss calculation, we need to determine a single target label for the entire grid
                target = output_grid.view(-1).mode().values.unsqueeze(0).long()
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                train_correct += (output.argmax(1) == target).sum().item()
                total_train_samples += 1
        
        avg_train_loss = total_train_loss / total_train_samples
        train_accuracy = train_correct / total_train_samples
        
        # Evaluation
        model.eval()
        total_eval_loss = 0
        eval_correct = 0
        total_eval_samples = 0
        
        with torch.no_grad():
            for task in tqdm(eval_tasks.values(), desc=f"Epoch {epoch+1}/{epochs} - Evaluation"):
                for input_grid, output_grid in task.test:
                    input_grid = input_grid.to(device).float()
                    output_grid = output_grid.to(device)
                    
                    output = model(input_grid.unsqueeze(0))  # Add batch dimension
                    
                    # Determine a single target label for the entire grid
                    target = output_grid.view(-1).mode().values.unsqueeze(0).long()
                    
                    loss = criterion(output, target)
                    
                    total_eval_loss += loss.item()
                    eval_correct += (output.argmax(1) == target).sum().item()
                    total_eval_samples += 1
        
        avg_eval_loss = total_eval_loss / total_eval_samples
        eval_accuracy = eval_correct / total_eval_samples
        
        # Update learning rate
        new_lr = model.update_lr(avg_eval_loss)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Model modifications
        if epoch % 5 == 0:
            if eval_accuracy < 0.85 and model.can_grow(stats):
                model.grow()
            elif eval_accuracy >= 0.85:
                for module in model.task_modules:
                    module.prune()
                    module.reset_weights()
        
        # Update and log statistics
        stats.update()
        stats.log_to_wandb()
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Eval Loss: {avg_eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}")
        print(f"Learning Rate: {new_lr:.6f}")
        print("-" * 50)
        
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'eval_loss': avg_eval_loss,
            'eval_accuracy': eval_accuracy,
            'learning_rate': new_lr,
        })
        
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_model = model.state_dict()
        
        if epoch % 100 == 0:
            save_checkpoint(model, optimizer, epoch, avg_eval_loss)
    
    return best_model, {
        'final_train_loss': avg_train_loss,
        'final_train_accuracy': train_accuracy,
        'final_eval_loss': avg_eval_loss,
        'final_eval_accuracy': eval_accuracy,
        'best_eval_loss': best_eval_loss,
        'model_size_mb': model.get_model_size(),
    }

def save_checkpoint(model, optimizer, epoch, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_structure': {
            'input_channels': model.input_channels,
            'num_classes': model.out_features,
            'num_modules': model.num_modules,
            'max_size_mb': model.max_size_mb,
            'task_modules': [
                {
                    'hidden_channels': module.hidden_channels,
                    'max_hidden_channels': module.max_hidden_channels,
                    'growth_factor': module.growth_factor
                }
                for module in model.task_modules
            ]
        }
    }
    torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pth")

def load_checkpoint(checkpoint_path, model_class):
    checkpoint = torch.load(checkpoint_path)
    model_structure = checkpoint['model_structure']
    
    model = model_class(
        input_channels=model_structure['input_channels'],
        num_classes=model_structure['num_classes'],
        num_modules=model_structure['num_modules'],
        max_size_mb=model_structure['max_size_mb']
    )
    
    # Reconstruct task modules
    for i, module_info in enumerate(model_structure['task_modules']):
        model.task_modules[i].hidden_channels = module_info['hidden_channels']
        model.task_modules[i].max_hidden_channels = module_info['max_hidden_channels']
        model.task_modules[i].growth_factor = module_info['growth_factor']
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss

def main():
    print_cuda_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()
    
    base_path = r'/root/projects/Ego/arc_challenge'
    train_challenge_file = os.path.join(base_path, 'arc-agi_training_challenges.json')
    train_solution_file = os.path.join(base_path, 'arc-agi_training_solutions.json')
    eval_challenge_file = os.path.join(base_path, 'arc-agi_evaluation_challenges.json')
    eval_solution_file = os.path.join(base_path, 'arc-agi_evaluation_solutions.json')
    
    print("Loading training data...")
    train_tasks = load_arc_data(train_challenge_file, train_solution_file)
    print("Loading evaluation data...")
    eval_tasks = load_arc_data(eval_challenge_file, eval_solution_file)
    
    wandb.login()
    
    config = {
        'input_channels': 1,    # 2D grid input
        'num_classes': 10,      # 0-9 possible values
        'num_modules': 3,
        'max_size_mb': 100
    }
    
    wandb.init(
        project="arc-adaptive-network",
        config=config
    )
    
    
    model = AdaptiveNetworkWithCorrelation(input_channels=1, num_classes=10, num_modules=10, max_size_mb=1000).to(device)
    trained_model, results = train_and_evaluate(model, train_tasks, eval_tasks, epochs=100000)

    wandb.log(results)
    torch.save(trained_model.state_dict(), "final_model.pth")
    wandb.finish()

if __name__ == "__main__":
    main()