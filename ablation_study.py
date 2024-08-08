import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
import json
from data_utils import ARCDataset, ARCTask
from utils import get_device, print_cuda_info
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from ego import SelfAdaptingModificationScheduler, SelfModifyingNetwork, AdaptiveEarlyStopping, AdaptiveLRScheduler
import copy



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


def run_ablation_study(model_class, train_dataset, eval_dataset, base_config, features_to_ablate, epochs=1000):
    results = {}
    device = get_device()
    
    # Run baseline model
    base_config['device'] = device
    baseline_model = model_class(**base_config).to(device)
    baseline_performance = train_and_evaluate(baseline_model, train_dataset, eval_dataset, epochs)
    results['baseline'] = baseline_performance
    
    # Run ablation for each feature
    for feature in features_to_ablate:
        ablated_config = base_config.copy()
        ablated_config[feature] = False
        ablated_model = model_class(**ablated_config).to(device)
        ablated_performance = train_and_evaluate(ablated_model, train_dataset, eval_dataset, epochs)
        results[f'ablated_{feature}'] = ablated_performance
    
    return results

def train_and_evaluate(model, train_tasks, eval_tasks, epochs):
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = AdaptiveLRScheduler(optimizer, mode='min', factor=0.5, patience=5, 
                                    threshold=1e-4, threshold_mode='rel', cooldown=0, 
                                    min_lr=1e-6, max_lr=1e-2, increase_factor=1.2, verbose=True)
    modification_scheduler = SelfAdaptingModificationScheduler(model, initial_frequency=0.1, min_frequency=0.01, max_frequency=0.5)
    early_stopping = AdaptiveEarlyStopping(patience=100, modification_delay=20)

    best_eval_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        modifications = []  # Initialize modifications at the start of each epoch
        model.train()
        total_train_loss = 0
        num_train_samples = 0
        
        for task in tqdm(train_tasks.values(), desc=f"Epoch {epoch+1}/{epochs}"):
            for input_grid, output_grid in task.train:
                input_grid = input_grid.unsqueeze(0).to(device)
                output_grid = output_grid.to(device)
                
                optimizer.zero_grad()
                content_pred = model(input_grid)
                
                content_pred = F.interpolate(content_pred, size=output_grid.shape, mode='nearest')
                content_pred_flat = content_pred.permute(0, 2, 3, 1).contiguous().view(-1, content_pred.size(1))
                output_grid_flat = output_grid.long().view(-1)
                
                num_classes = content_pred.size(1)
                if output_grid_flat.max() >= num_classes:
                    output_grid_flat = torch.clamp(output_grid_flat, 0, num_classes - 1)
                
                loss = torch.nn.functional.cross_entropy(content_pred_flat, output_grid_flat)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_train_loss += loss.item()
                num_train_samples += 1
        
        avg_train_loss = total_train_loss / num_train_samples if num_train_samples > 0 else float('inf')
        
        model.eval()
        total_eval_loss = 0
        num_eval_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for task in eval_tasks.values():
                for input_grid, output_grid in task.test:
                    input_grid = input_grid.unsqueeze(0).to(device)
                    output_grid = output_grid.to(device)
                    
                    content_pred = model(input_grid)
                    content_pred = F.interpolate(content_pred, size=output_grid.shape, mode='nearest')
                    content_pred_flat = content_pred.permute(0, 2, 3, 1).contiguous().view(-1, content_pred.size(1))
                    output_grid_flat = output_grid.long().view(-1)
                    
                    num_classes = content_pred.size(1)
                    if output_grid_flat.max() >= num_classes:
                        output_grid_flat = torch.clamp(output_grid_flat, 0, num_classes - 1)
                    
                    loss = torch.nn.functional.cross_entropy(content_pred_flat, output_grid_flat)
                    total_eval_loss += loss.item()
                    num_eval_samples += 1
                    
                    predicted = content_pred.argmax(dim=1).view(output_grid.shape)
                    correct_predictions += (predicted == output_grid).all().item()
        
        accuracy = correct_predictions / num_eval_samples if num_eval_samples > 0 else 0        
        avg_eval_loss = total_eval_loss / num_eval_samples if num_eval_samples > 0 else float('inf')
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_eval_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        modification_scheduler.update(avg_train_loss, avg_eval_loss)
        
        if modification_scheduler.should_modify():
            modifications = model.modify(avg_train_loss, avg_eval_loss)
            if modifications:
                print(f"Epoch {epoch+1}: Model modifications:")
                for mod in modifications:
                    print(f"  - {mod}")
                modification_scheduler.record_modification()
                
                # Re-create optimizer and scheduler after modifications
                optimizer = torch.optim.Adam(model.parameters(), lr=new_lr, weight_decay=1e-5)
                scheduler = AdaptiveLRScheduler(optimizer, mode='min', factor=0.5, patience=5, 
                                                threshold=1e-4, threshold_mode='rel', cooldown=0, 
                                                min_lr=1e-6, max_lr=1e-2, increase_factor=1.2, verbose=True)

        stats = modification_scheduler.get_stats()
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Modification Frequency: {stats['frequency']:.4f}, Parameters: {stats['parameters']}")
        print(f"Learning Rate: {new_lr:.6f}")
        if old_lr != new_lr:
            print(f"Learning rate changed: {old_lr:.6f} -> {new_lr:.6f}")
        print("-" * 50)
        
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'eval_loss': avg_eval_loss,
            'loss_gap': avg_eval_loss - avg_train_loss,
            'accuracy': accuracy,
            'modification_frequency': stats['frequency'],
            'total_modifications': stats['modifications'],
            'parameter_count': stats['parameters'],
            'learning_rate': new_lr,
            'lr_change': (new_lr - old_lr) / old_lr,
        })
        
        if modifications:
            for i, mod in enumerate(modifications):
                wandb.log({f'modification_{i}': mod})

        should_stop, best_model = early_stopping(model, epoch, avg_eval_loss, modifications)
        if should_stop:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Use the best model for final evaluation
    model = best_model
    
    return {
        'final_train_loss': avg_train_loss,
        'final_eval_loss': avg_eval_loss,
        'best_eval_loss': best_eval_loss,
        'final_accuracy': accuracy,
        'model_size_mb': model.get_model_size(),
        'num_modifications': stats['modifications'],
        'final_parameter_count': stats['parameters']
    }

def main():
    from ego import SelfModifyingNetwork  # Import your model class
    
    print_cuda_info()
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
    
    base_config = {
        'input_channels': 10,  # ARC typically uses values 0-9
        'initial_hidden_channels': [64, 128],
        'max_size_mb': 100,
        'enable_layer_addition': True,
        'enable_activation_modification': True,
        'enable_forward_modification': True
    }
    
    features_to_ablate = ['enable_layer_addition', 'enable_activation_modification', 'enable_forward_modification']
    
    wandb.init(
        project="arc-self-modifying-network-ablation",
        config={
            "initial_hidden_channels": base_config['initial_hidden_channels'],
            "max_size_mb": base_config['max_size_mb'],
            "enable_layer_addition": base_config['enable_layer_addition'],
            "enable_activation_modification": base_config['enable_activation_modification'],
            "enable_forward_modification": base_config['enable_forward_modification'],
        }
    )    
    results = run_ablation_study(SelfModifyingNetwork, train_tasks, eval_tasks, base_config, features_to_ablate)
    
    # Log results to wandb
    #for study_name, performance in results.items():
        #wandb.log({f"{study_name}_{k}": v for k, v in performance.items()})
    #wandb.finish()

if __name__ == "__main__":
    main()