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
from ego import SelfAdaptingModificationScheduler, SelfModifyingNetwork, AdaptiveEarlyStopping
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


def run_ablation_study(model_class, train_dataset, eval_dataset, base_config, features_to_ablate, epochs=100):
    results = {}
    
    # Run baseline model
    baseline_model = model_class(**base_config)
    baseline_performance = train_and_evaluate(baseline_model, train_dataset, eval_dataset, epochs)
    results['baseline'] = baseline_performance
    
    # Run ablation for each feature
    for feature in features_to_ablate:
        ablated_config = base_config.copy()
        ablated_config[feature] = False
        ablated_model = model_class(**ablated_config)
        ablated_performance = train_and_evaluate(ablated_model, train_dataset, eval_dataset, epochs)
        results[f'ablated_{feature}'] = ablated_performance
    
    return results

def train_and_evaluate(model, train_tasks, eval_tasks, epochs):
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    modification_scheduler = SelfAdaptingModificationScheduler(model, initial_frequency=0.1, min_frequency=0.01, max_frequency=0.2)
    early_stopping = AdaptiveEarlyStopping(patience=20, modification_delay=20)

    best_eval_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
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
        
        modification_scheduler.update(avg_eval_loss)
        
        if modification_scheduler.should_modify():
            modifications = model.modify(avg_train_loss, avg_eval_loss)
        else:
            modifications = []

        should_stop, best_model = early_stopping(model, epoch, avg_eval_loss, modifications)
        
        stats = modification_scheduler.get_stats()
        
        # Terminal logging
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Modification Frequency: {stats['frequency']:.4f}, Parameters: {stats['parameters']}")
        print(f"Learning Rate: {new_lr:.6f}")
        if modifications:
            print("Modifications made:")
            for mod in modifications:
                print(f"- {mod}")
        if old_lr != new_lr:
            print(f"Learning rate changed: {old_lr:.6f} -> {new_lr:.6f}")
        print("-" * 50)
        
        # wandb logging
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'eval_loss': avg_eval_loss,
            'accuracy': accuracy,
            'modification_frequency': stats['frequency'],
            'cooldown_period': stats['cooldown'],
            'modification_threshold': stats['threshold'],
            'total_modifications': stats['modifications'],
            'parameter_count': stats['parameters'],
            'learning_rate': new_lr,
            'model_size_mb': model.get_model_size(),
            'num_encoder_layers': len(model.encoder),
            'num_decoder_layers': len(model.decoder),
            'current_activation': model.current_activation,
        })
        
        if modifications:
            wandb.log({f'modification_{i}': mod for i, mod in enumerate(modifications)})
        
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
    
    base_path = r'C:\Users\Clemspace\Mistral\EGO\arc_challenge'
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
        'max_size_mb': 50,
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