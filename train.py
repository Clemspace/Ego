# File: train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR
from tqdm import tqdm
import wandb
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

def augment_grid(grid):
    transforms = nn.Sequential(
        RandomRotation(degrees=(-90, 90)),
        RandomHorizontalFlip(),
        RandomVerticalFlip()
    )
    return transforms(grid.unsqueeze(0)).squeeze(0)

def train_model(model, train_dataset, eval_dataset, epochs=1000, modify_every=10, accumulation_steps=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    initial_lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    
    # Set initial_lr for each parameter group
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = initial_lr
        
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler()

    wandb.init(project="arc-solver", config={
        "epochs": epochs,
        "modify_every": modify_every,
        "device": device.type,
        "initial_lr": initial_lr,
        "initial_param_count": model.count_parameters()
    })
    
    wandb.watch(model, log="all", log_freq=100)

    best_eval_loss = float('inf')
    best_epoch = 0
    plateau_length = 0
    max_plateau_length = 100
    
    previous_loss = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        
        train_completion_rates = {k: 0.0 for k in ["25%_complete", "50%_complete", "75%_complete", "100%_complete", "size_accuracy"]}
        
        for batch_idx, (subtask_inputs, subtask_outputs) in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1}/{epochs}")):
            global_step = epoch * len(train_dataset) + batch_idx
            
            subtask_content_preds = []
            subtask_size_preds = []
            
            with torch.cuda.amp.autocast():
                for inp in subtask_inputs:
                    if inp is not None:
                        inp = inp.unsqueeze(0).to(device)
                        inp = (inp - inp.mean()) / (inp.std() + 1e-8)  # Normalize input
                        content_pred, size_pred = model(inp)
                        subtask_content_preds.append(content_pred.squeeze(0))
                        subtask_size_preds.append(size_pred)
                    else:
                        subtask_content_preds.append(None)
                        subtask_size_preds.append(None)
                
                loss = custom_loss(subtask_content_preds, subtask_size_preds, subtask_outputs, model.content_weight, model.size_weight, model)

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print(f"Gradient norm is {total_norm}, skipping this batch.")
                    optimizer.zero_grad()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_train_loss += loss.item()
                num_batches += 1
                
                batch_completion_rates = calculate_task_completion(subtask_content_preds, subtask_size_preds, subtask_outputs)
                for k, v in batch_completion_rates.items():
                    train_completion_rates[k] += v

        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')
        avg_train_completion_rates = {k: v / len(train_dataset) for k, v in train_completion_rates.items()}
        
        # Evaluation
        model.eval()
        total_eval_loss = 0
        eval_completion_rates = {k: 0.0 for k in ["25%_complete", "50%_complete", "75%_complete", "100%_complete", "size_accuracy"]}
        
        with torch.no_grad():
            for subtask_inputs, subtask_outputs in eval_dataset:
                subtask_content_preds = []
                subtask_size_preds = []
                
                for inp in subtask_inputs:
                    if inp is not None:
                        inp = inp.unsqueeze(0).to(device)
                        inp = (inp - inp.mean()) / (inp.std() + 1e-8)  # Normalize input
                        content_pred, size_pred = model(inp)
                        subtask_content_preds.append(content_pred.squeeze(0))
                        subtask_size_preds.append(size_pred)
                    else:
                        subtask_content_preds.append(None)
                        subtask_size_preds.append(None)
                
                loss = custom_loss(subtask_content_preds, subtask_size_preds, subtask_outputs, model.content_weight, model.size_weight, model)
                total_eval_loss += loss.item()
                
                batch_completion_rates = calculate_task_completion(subtask_content_preds, subtask_size_preds, subtask_outputs)
                for k, v in batch_completion_rates.items():
                    eval_completion_rates[k] += v
        
        avg_eval_loss = total_eval_loss / len(eval_dataset)
        avg_eval_completion_rates = {k: v / len(eval_dataset) for k, v in eval_completion_rates.items()}
        
        # Adjust learning rate at the end of the epoch
        if previous_loss is not None:
            adjustment_factor = model.adjust_learning_rate(optimizer, avg_eval_loss, previous_loss)
            wandb.log({"lr_adjustment_factor": adjustment_factor})
            
        previous_loss = avg_eval_loss

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")
        print("Train Completion Rates:", avg_train_completion_rates)
        print("Eval Completion Rates:", avg_eval_completion_rates)
        print(f"Current learning rate: {current_lr}")
        print(f"Current parameter count: {model.count_parameters()}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "eval_loss": avg_eval_loss,
            "learning_rate": current_lr,
            "gradient_norm": total_norm,
            "parameter_count": model.count_parameters(),
            **{f"train_{k}": v for k, v in avg_train_completion_rates.items()},
            **{f"eval_{k}": v for k, v in avg_eval_completion_rates.items()}
        })
        
        if (epoch + 1) % modify_every == 0:
            modifications = model.modify(avg_eval_loss)
            print("Model modifications:", modifications)
            # Update the optimizer's learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = model.learning_rate.item()
            
            # Log parameter count changes
            if model.param_count_history:
                param_count_change = model.param_count_history[-1] - model.param_count_history[-2]
                wandb.log({"parameter_count_change": param_count_change})
        
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_epoch = epoch
            plateau_length = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("New best model saved!")
        else:
            plateau_length += 1
            
        if plateau_length >= max_plateau_length:
            print(f"No improvement for {max_plateau_length} epochs. Stopping training.")
            print(f"Best performance was at epoch {best_epoch + 1}")
            break
    
    wandb.finish()
    return model

def custom_loss(content_preds, size_preds, targets, content_weight, size_weight, model, l2_lambda=1e-4, epsilon=1e-8):
    total_loss = 0
    num_valid = 0
    
    for content_pred, size_pred, target in zip(content_preds, size_preds, targets):
        if target is None or content_pred is None or size_pred is None:
            continue
        
        num_valid += 1
        true_h, true_w = target.shape
        
        content_pred_resized = F.interpolate(content_pred.unsqueeze(0), size=(true_h, true_w), mode='bilinear', align_corners=False)
        
        content_loss = F.cross_entropy(content_pred_resized, target.unsqueeze(0).long(), reduction='sum')
        
        pred_h, pred_w = size_pred
        true_size = torch.tensor([true_h, true_w], device=content_pred.device, dtype=torch.float)
        pred_size = torch.tensor([pred_h, pred_w], device=content_pred.device, dtype=torch.float)
        size_loss = F.mse_loss(pred_size, true_size, reduction='sum')
        
        loss = content_weight * content_loss + size_weight * size_loss
        
        total_loss += torch.log1p(loss)
    
    avg_loss = total_loss / max(num_valid, 1)
    
    l2_reg = sum(param.pow(2.0).sum() for param in model.parameters())
    
    final_loss = avg_loss + l2_lambda * l2_reg
    
    final_loss = torch.clamp(final_loss, min=epsilon, max=1e6)

    return final_loss

def calculate_task_completion(content_preds, size_preds, targets, thresholds=[0.25, 0.5, 0.75, 1.0]):
    completion_rates = {f"{int(threshold*100)}%_complete": 0.0 for threshold in thresholds}
    size_accuracy = 0.0
    total_valid = 0

    for content_pred, size_pred, target in zip(content_preds, size_preds, targets):
        if target is None:
            continue

        total_valid += 1
        true_h, true_w = target.shape
        pred_h, pred_w = size_pred

        size_tolerance = 0.1
        if (abs(pred_h - true_h) <= true_h * size_tolerance) and (abs(pred_w - true_w) <= true_w * size_tolerance):
            size_accuracy += 1

        content_pred_resized = F.interpolate(content_pred.unsqueeze(0), size=(true_h, true_w), mode='bilinear', align_corners=False)
        
        pred_content = content_pred_resized.squeeze(0).argmax(dim=0)
        pixel_accuracy = (pred_content == target).float().mean().item()

        for threshold in thresholds:
            if pixel_accuracy >= threshold:
                completion_rates[f"{int(threshold*100)}%_complete"] += 1

    if total_valid > 0:
        for key in completion_rates:
            completion_rates[key] /= total_valid
        size_accuracy /= total_valid
    
    completion_rates["size_accuracy"] = size_accuracy
    return completion_rates

# Commented out ablated models
"""
# Full model
model_full = SelfModifyingNetwork(input_channels, initial_hidden_channels)

# Model without layer normalization
model_no_ln = SelfModifyingNetwork(input_channels, initial_hidden_channels, use_layer_norm=False)

# Model without ELU activation
model_no_elu = SelfModifyingNetwork(input_channels, initial_hidden_channels, use_elu=False)

# Model without feedback system
model_no_feedback = SelfModifyingNetwork(input_channels, initial_hidden_channels, use_feedback=False)

# Model without architecture modification
model_no_arch_mod = SelfModifyingNetwork(input_channels, initial_hidden_channels, allow_architecture_mod=False)
"""