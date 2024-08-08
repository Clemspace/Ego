import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import wandb

def train_model(model, train_dataset, eval_dataset, epochs=1000, modify_every=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    wandb.init(project="self-modifying-arc-solver", config={
        "epochs": epochs,
        "modify_every": modify_every,
        "max_size_mb": model.max_size_mb
    })

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataset)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataset)

        # Evaluation
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for inputs, targets in eval_dataset:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                total_eval_loss += loss.item()

        avg_eval_loss = total_eval_loss / len(eval_dataset)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "eval_loss": avg_eval_loss,
            "model_size_mb": model.get_model_size()
        })

        # Self-modification
        if (epoch + 1) % modify_every == 0:
            modifications = model.modify(avg_eval_loss)
            print("Model modifications:", modifications)
            wandb.log({"modifications": modifications})

        # Check if model size exceeds limit
        if model.get_model_size() > model.max_size_mb:
            print(f"Warning: Model size ({model.get_model_size():.2f} MB) exceeds the limit ({model.max_size_mb} MB).")
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