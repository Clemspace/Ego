import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import random
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model import SelfModifyingNetwork
from data_utils import ARCTask, ARCDataset
from train import train_model, custom_loss, calculate_task_completion
from utils import get_device, print_cuda_info

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

def evaluate_on_task(model, task, device):
    model.eval()
    correct_content = 0
    correct_size = 0
    total = 0
    skipped = 0
    
    with torch.no_grad():
        for input_grid, output_grid in task.test:
            input_grid = input_grid.unsqueeze(0).unsqueeze(0).to(device)
            output_grid = output_grid.to(device)
            input_grid = (input_grid - input_grid.mean()) / (input_grid.std() + 1e-8)  # Normalize input
            content_pred, size_pred = model(input_grid)
            
            pred_h, pred_w = size_pred
            true_h, true_w = output_grid.shape
            size_tolerance = 0.1  # 10% tolerance
            if (abs(pred_h - true_h) <= true_h * size_tolerance) and (abs(pred_w - true_w) <= true_w * size_tolerance):
                correct_size += 1
            
            content_pred = F.interpolate(content_pred.unsqueeze(0), size=(true_h, true_w), mode='bilinear', align_corners=False)
            content_pred = content_pred.squeeze(0).argmax(dim=0)
            
            if content_pred.shape == output_grid.shape:
                correct_content += (content_pred == output_grid).all().item()
            else:
                print(f"Skipping comparison due to size mismatch: pred {content_pred.shape}, true {output_grid.shape}")
                skipped += 1
            
            total += 1
    
    content_accuracy = correct_content / total if total > 0 else 0
    size_accuracy = correct_size / total if total > 0 else 0
    
    return {
        'content_accuracy': content_accuracy,
        'size_accuracy': size_accuracy,
        'total_cases': total,
        'evaluated_cases': total - skipped,
        'skipped_cases': skipped
    }

def visualize_random_entries(model, train_dataset, eval_dataset, device, num_samples=3):
    model.eval()

    def visualize_sample(dataset, title):
        index = random.randint(0, len(dataset) - 1)
        subtask_inputs, subtask_outputs = dataset[index]

        with torch.no_grad():
            for inp, out in zip(subtask_inputs, subtask_outputs):
                if inp is not None and out is not None:
                    input_tensor = inp.unsqueeze(0).unsqueeze(0).to(device)
                    input_tensor = (input_tensor - input_tensor.mean()) / (input_tensor.std() + 1e-8)  # Normalize input
                    content_pred, size_pred = model(input_tensor)
                    pred_h, pred_w = size_pred
                    predicted_output = content_pred.argmax(dim=1).squeeze().cpu()[:pred_h, :pred_w]

                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    ax1.imshow(inp.cpu(), cmap='viridis')
                    ax1.set_title('Input')
                    ax2.imshow(out.cpu(), cmap='viridis')
                    ax2.set_title(f'Ground Truth ({out.shape[0]}x{out.shape[1]})')
                    ax3.imshow(predicted_output, cmap='viridis')
                    ax3.set_title(f'Model Output ({pred_h}x{pred_w})')
                    plt.suptitle(title)
                    wandb.log({title: wandb.Image(plt)})
                    plt.close()
                    break  # Only visualize the first non-None input-output pair

    for i in range(num_samples):
        visualize_sample(train_dataset, f"Training Sample {i+1}")
        visualize_sample(eval_dataset, f"Evaluation Sample {i+1}")

def main():
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

    train_dataset = ARCDataset(list(train_tasks.values()))
    eval_dataset = ARCDataset(list(eval_tasks.values()))

    model = SelfModifyingNetwork(input_channels=1, initial_hidden_channels=[64, 128, 256, 512], max_output_size=30)
    model = model.to(device)
    
    wandb.init(project="arc-solver")
    wandb.watch(model, log="all")
    initial_architecture = model.get_architecture_summary()
    wandb.log({"initial_architecture": wandb.Table(data=[[i, layer] for i, layer in enumerate(initial_architecture)], 
                                                   columns=["Layer", "Description"])})

    epochs = 1000
    modify_every = 10

    print("Starting training...")
    train_model(model, train_dataset, eval_dataset, epochs=epochs, modify_every=modify_every)

    print("Evaluating model on evaluation set...")
    content_correct = 0
    size_correct = 0
    total = 0
    for task in eval_tasks.values():
        evaluation_results = evaluate_on_task(model, task, device)
        if evaluation_results['content_accuracy'] is not None:
            content_correct += evaluation_results['content_accuracy'] * evaluation_results['evaluated_cases']
            size_correct += evaluation_results['size_accuracy'] * evaluation_results['evaluated_cases']
            total += evaluation_results['evaluated_cases']

    if total > 0:
        content_accuracy = content_correct / total
        size_accuracy = size_correct / total
        print(f"Overall content accuracy on evaluation set: {content_accuracy:.4f}")
        print(f"Overall size accuracy on evaluation set: {size_accuracy:.4f}")
        wandb.log({"final_content_accuracy": content_accuracy, "final_size_accuracy": size_accuracy})
    else:
        print("No valid evaluation cases found.")

    print("Visualizing random entries...")
    visualize_random_entries(model, train_dataset, eval_dataset, device, num_samples=3)

    torch.save(model.state_dict(), 'arc_solver_model.pth')
    wandb.save('arc_solver_model.pth')
    print("Model saved successfully.")

    wandb.finish()

if __name__ == "__main__":
    main()