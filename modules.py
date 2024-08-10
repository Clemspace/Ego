import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import wandb
from collections import OrderedDict
import wandb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_size):
    def hook_fn(module, input, output):
        if isinstance(module, nn.Conv2d):
            # Ensure input is 4D (batch_size, channels, height, width)
            if input[0].dim() == 3:
                input = (input[0].unsqueeze(1),)
            elif input[0].dim() == 2:
                input = (input[0].unsqueeze(0).unsqueeze(0),)
            
            batch_size, input_channels, input_height, input_width = input[0].size()
            output_channels, output_height, output_width = output.size()[1:]
            
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (input_channels // module.groups)
            flops = kernel_ops * output_channels * output_height * output_width * batch_size
            
            module.__flops__ = flops
        elif isinstance(module, nn.Linear):
            batch_size = input[0].size(0)
            flops = batch_size * input[0].size(1) * output.size(-1)
            module.__flops__ = flops

    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))

    device = next(model.parameters()).device
    input = torch.randn(1, *input_size).to(device)
    
    # Ensure input is 4D
    if input.dim() == 2:
        input = input.unsqueeze(0).unsqueeze(0)
    elif input.dim() == 3:
        input = input.unsqueeze(1)
    
    model(input)

    total_flops = 0
    for module in model.modules():
        if hasattr(module, '__flops__'):
            total_flops += module.__flops__

    for hook in hooks:
        hook.remove()

    return total_flops

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

class ModelStatistics:
    def __init__(self, model, input_size):
        self.model = model
        self.input_size = input_size
        self.update()

    def update(self):
        self.total_params = count_parameters(self.model)
        self.total_flops = count_flops(self.model, self.input_size)
        self.model_size_mb = get_model_size(self.model)
        
        self.module_stats = []
        for i, module in enumerate(self.model.task_modules):
            module_params = count_parameters(module)
            module_flops = count_flops(module, self.input_size)
            module_size_mb = get_model_size(module)
            self.module_stats.append({
                'module_id': i,
                'params': module_params,
                'flops': module_flops,
                'size_mb': module_size_mb
            })
            
    def log_to_wandb(self):
        wandb.log({
            'total_params': self.total_params,
            'total_flops': self.total_flops,
            'model_size_mb': self.model_size_mb
        })
        for stats in self.module_stats:
            wandb.log({
                f'module_{stats["module_id"]}_params': stats['params'],
                f'module_{stats["module_id"]}_flops': stats['flops'],
                f'module_{stats["module_id"]}_size_mb': stats['size_mb']
            })

class MetaLearner(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = next(model.parameters()).device

    def clone_model_with_theta(self, theta):
        model_clone = type(self.model)(
            self.model.input_channels, 
            self.model.classifier[-1].out_features,
            num_modules=len(self.model.task_modules),
            max_size_mb=self.model.max_size_mb
        ).to(self.device)
        
        # Load parameters
        model_clone.load_state_dict(theta, strict=False)
        
        # Copy batch norm statistics
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                clone_module = dict(model_clone.named_modules())[name]
                clone_module.running_mean = module.running_mean.clone().to(self.device)
                clone_module.running_var = module.running_var.clone().to(self.device)
                clone_module.num_batches_tracked = module.num_batches_tracked.clone().to(self.device)
        
        return model_clone

    def unroll_loop(self, support_loader, query_loader, steps=5):
        create_graph = True
        theta = OrderedDict((name, param.clone().to(self.device)) for name, param in self.model.named_parameters())
        
        # Get the learning rate value
        lr = self.model.lr_module.lr.item()
        
        for _ in range(steps):
            support_loss = 0
            for batch in support_loader:
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                model_clone = self.clone_model_with_theta(theta)
                output = model_clone(data)
                
                # Reshape output and target
                output = output.view(-1, output.size(-1))
                target = target.view(-1)
                
                # Ensure output and target have the same first dimension
                if output.size(0) != target.size(0):
                    min_size = min(output.size(0), target.size(0))
                    output = output[:min_size]
                    target = target[:min_size]
                
                loss = F.cross_entropy(output, target.long())
                support_loss += loss
            
            grads = torch.autograd.grad(support_loss, theta.values(), create_graph=create_graph, allow_unused=True)
            
            # Handle None gradients
            grads = [torch.zeros_like(param) if grad is None else grad for grad, param in zip(grads, theta.values())]
            
            theta = OrderedDict(
                (name, param - lr * grad)
                for ((name, param), grad) in zip(theta.items(), grads)
            )

        query_loss = 0
        for batch in query_loader:
            data, target = batch
            data, target = data.to(self.device), target.to(self.device)
            model_clone = self.clone_model_with_theta(theta)
            output = model_clone(data)
            
            # Reshape output and target
            output = output.view(-1, output.size(-1))
            target = target.view(-1)
            
            # Ensure output and target have the same first dimension
            if output.size(0) != target.size(0):
                min_size = min(output.size(0), target.size(0))
                output = output[:min_size]
                target = target[:min_size]
            
            loss = F.cross_entropy(output, target.long())
            query_loss += loss

        return query_loss

    def meta_update(self, support_loader, query_loader):
        query_loss = self.unroll_loop(support_loader, query_loader)
        self.meta_optimizer.zero_grad()
        query_loss.backward()
        self.meta_optimizer.step()
        return query_loss.item()


class AdaptiveLRModule(nn.Module):
    def __init__(self, initial_lr=0.001, min_lr=1e-6, max_lr=0.1, 
                 increase_factor=1.2, decrease_factor=0.8, 
                 patience=5, cooldown=10):
        super().__init__()
        self.lr = nn.Parameter(torch.tensor(initial_lr))
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.patience = patience
        self.cooldown = cooldown
        
        self.best_loss = float('inf')
        self.bad_epochs = 0
        self.cooldown_counter = 0
        self.history = []

    def forward(self, loss):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.lr.item()

        if loss < self.best_loss:
            self.best_loss = loss
            self.bad_epochs = 0
            if self.lr < self.max_lr:
                self.lr.data = torch.clamp(self.lr * self.increase_factor, max=self.max_lr)
        else:
            self.bad_epochs += 1

        if self.bad_epochs > self.patience:
            self.lr.data = torch.clamp(self.lr * self.decrease_factor, min=self.min_lr)
            self.bad_epochs = 0
            self.cooldown_counter = self.cooldown

        self.history.append(self.lr.item())
        return self.lr.item()

    def reset(self):
        self.best_loss = float('inf')
        self.bad_epochs = 0
        self.cooldown_counter = 0

class TaskCorrelationLearner(nn.Module):
    def __init__(self, num_modules):
        super().__init__()
        self.correlation_matrix = nn.Parameter(torch.eye(num_modules))
        
    def forward(self, module_outputs):
        # module_outputs shape: (batch_size, num_modules, channels, height, width)
        corr = F.softmax(self.correlation_matrix, dim=1)
        
        # Reshape for matrix multiplication
        batch_size, num_modules, channels, height, width = module_outputs.shape
        reshaped_outputs = module_outputs.view(batch_size, num_modules, -1)
        
        # Apply correlation
        correlated = torch.matmul(corr, reshaped_outputs)
        
        # Reshape back to original dimensions
        return correlated.view(batch_size, num_modules, channels, height, width)

    def expand(self, new_size):
        old_size = self.correlation_matrix.size(0)
        if new_size > old_size:
            new_matrix = torch.eye(new_size, device=self.correlation_matrix.device)
            new_matrix[:old_size, :old_size] = self.correlation_matrix
            self.correlation_matrix = nn.Parameter(new_matrix)
    
class AdaptiveTaskModule(nn.Module):
    def __init__(self, input_channels, initial_hidden_channels=64, max_hidden_channels=256, growth_factor=1.5):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = initial_hidden_channels
        self.max_hidden_channels = max_hidden_channels
        self.growth_factor = growth_factor

        self.conv1 = nn.Conv2d(input_channels, self.hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.hidden_channels)
        self.bn2 = nn.BatchNorm2d(self.hidden_channels)
        
        self.mask = None
        self.original_weights = None
        
        self.lr_module = AdaptiveLRModule()

    def forward(self, x):
        # Ensure input is 4D (batch_size, channels, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

    def grow(self):
        if self.hidden_channels < self.max_hidden_channels:
            new_hidden_channels = min(int(self.hidden_channels * self.growth_factor), self.max_hidden_channels)
            self._expand_layer(self.conv1, new_hidden_channels, 1)  # Expand output channels of conv1
            self._expand_layer(self.conv2, new_hidden_channels, 2)  # Expand both input and output channels of conv2
            self._expand_bn(self.bn1, new_hidden_channels)
            self._expand_bn(self.bn2, new_hidden_channels)
            old_hidden_channels = self.hidden_channels
            self.hidden_channels = new_hidden_channels
            print(f"Grown from {old_hidden_channels} to {self.hidden_channels} channels")
            return True
        return False

    def _expand_layer(self, layer, new_channels, mode):
        device = layer.weight.device
        old_weight = layer.weight.data
        old_out_channels, old_in_channels = old_weight.shape[:2]

        if mode == 0:  # Expanding input channels only
            new_weight = torch.randn(old_out_channels, new_channels, *old_weight.shape[2:], device=device) * 0.02
            new_weight[:, :old_in_channels] = old_weight
            new_bias = layer.bias.data  # Bias doesn't change when only input channels expand
        elif mode == 1:  # Expanding output channels only
            new_weight = torch.randn(new_channels, old_in_channels, *old_weight.shape[2:], device=device) * 0.02
            new_weight[:old_out_channels] = old_weight
            new_bias = torch.zeros(new_channels, device=device)
            new_bias[:old_out_channels] = layer.bias.data
        else:  # Expanding both input and output channels
            new_weight = torch.randn(new_channels, new_channels, *old_weight.shape[2:], device=device) * 0.02
            new_weight[:old_out_channels, :old_in_channels] = old_weight
            new_bias = torch.zeros(new_channels, device=device)
            new_bias[:old_out_channels] = layer.bias.data

        layer.weight = nn.Parameter(new_weight)
        layer.bias = nn.Parameter(new_bias)
        
        if mode in [0, 2]:
            layer.in_channels = new_channels
        if mode in [1, 2]:
            layer.out_channels = new_channels

    def _expand_bn(self, bn, new_channels):
        device = bn.weight.device
        bn.num_features = new_channels
        bn.running_mean = torch.cat([bn.running_mean, torch.zeros(new_channels - bn.running_mean.shape[0], device=device)])
        bn.running_var = torch.cat([bn.running_var, torch.ones(new_channels - bn.running_var.shape[0], device=device)])
        bn.weight = nn.Parameter(torch.cat([bn.weight, torch.ones(new_channels - bn.weight.shape[0], device=device)]))
        bn.bias = nn.Parameter(torch.cat([bn.bias, torch.zeros(new_channels - bn.bias.shape[0], device=device)]))


    def _expand_attention(self, new_channels):
        self.attention = nn.MultiheadAttention(new_channels, 4)

    def prune(self, prune_ratio=0.2):
        with torch.no_grad():
            if self.mask is None:
                self.mask = {name: torch.ones_like(param) for name, param in self.named_parameters() if 'weight' in name}
                self.original_weights = {name: param.clone() for name, param in self.named_parameters() if 'weight' in name}

            for name, param in self.named_parameters():
                if 'weight' in name:
                    tensor = param.data
                    alive = self.mask[name].sum()
                    num_prune = int(alive * prune_ratio)
                    threshold = tensor.abs().view(-1).kthvalue(num_prune).values

                    new_mask = (tensor.abs() > threshold).float()
                    self.mask[name] = new_mask
                    param.data = self.original_weights[name] * new_mask

    def reset_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    param.data = self.original_weights[name] * self.mask[name]

    def update_lr(self, loss):
        return self.lr_module(loss)

class AdaptiveNetworkWithCorrelation(nn.Module):
    def __init__(self, input_channels, num_classes, num_modules=3, max_size_mb=100):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes  # Store num_classes as an attribute
        self.num_modules = num_modules
        self.task_modules = nn.ModuleList([AdaptiveTaskModule(input_channels) for _ in range(num_modules)])
        self.correlation_learner = TaskCorrelationLearner(num_modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = self._create_classifier(num_classes)
        self.max_size_mb = max_size_mb
        self.lr_module = AdaptiveLRModule()

    def _create_classifier(self, num_classes):
        current_hidden_channels = self.task_modules[0].hidden_channels
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_hidden_channels * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        module_outputs = torch.stack([module(x) for module in self.task_modules], dim=1)
        correlated_output = self.correlation_learner(module_outputs)
        x = correlated_output.sum(dim=1)
        x = self.adaptive_pool(x)
        return self.classifier(x)

    def update_lr(self, loss):
        return self.lr_module(loss)

    def can_grow(self, stats):
        return stats.model_size_mb < self.max_size_mb

    def grow(self):
        grew = False
        for module in self.task_modules:
            if module.grow():
                grew = True
        if grew:
            self._update_classifier()
            self.correlation_learner.expand(len(self.task_modules))
        return grew

    def _update_classifier(self):
        current_hidden_channels = self.task_modules[0].hidden_channels
        old_classifier = self.classifier
        self.classifier = self._create_classifier(old_classifier[-1].out_features)
        
        # Move the new classifier to the same device as the old one
        device = next(old_classifier.parameters()).device
        self.classifier = self.classifier.to(device)
        
        # Copy weights for the first linear layer, padding with zeros if necessary
        with torch.no_grad():
            old_weight = old_classifier[1].weight
            new_weight = self.classifier[1].weight
            min_dim = min(old_weight.size(1), new_weight.size(1))
            new_weight[:, :min_dim] = old_weight[:, :min_dim]
            
            # Copy weights and bias for the last linear layer
            self.classifier[-1].weight.copy_(old_classifier[-1].weight)
            self.classifier[-1].bias.copy_(old_classifier[-1].bias)

    def get_model_size(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def to(self, device):
        super().to(device)
        self.task_modules = nn.ModuleList([module.to(device) for module in self.task_modules])
        self.correlation_learner = self.correlation_learner.to(device)
        self.classifier = self.classifier.to(device)
        return self
    @property
    def out_features(self):
        return self.num_classes