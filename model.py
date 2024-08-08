import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DynamicLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        num_groups = 1
        for i in range(min(8, out_channels), 0, -1):
            if out_channels % i == 0:
                num_groups = i
                break
        
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.activation = nn.LeakyReLU(0.2)
        
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class SelfModifyingNetwork(nn.Module):
    def __init__(self, input_channels, initial_hidden_channels, max_output_size=30,
                 use_layer_norm=True, use_elu=True, use_feedback=True, allow_architecture_mod=True):
        super(SelfModifyingNetwork, self).__init__()
        self.input_channels = input_channels
        self.max_output_size = max_output_size
        self.use_layer_norm = use_layer_norm
        self.use_elu = use_elu
        self.use_feedback = use_feedback
        self.allow_architecture_mod = allow_architecture_mod

        self.encoder = nn.ModuleList([
            DynamicLayer(input_channels, initial_hidden_channels[0]),
            DynamicLayer(initial_hidden_channels[0], initial_hidden_channels[1]),
            DynamicLayer(initial_hidden_channels[1], initial_hidden_channels[2]),
            DynamicLayer(initial_hidden_channels[2], initial_hidden_channels[3])
        ])

        self.decoder = nn.ModuleList([
            DynamicLayer(initial_hidden_channels[3], 128),
            DynamicLayer(128, 64),
            DynamicLayer(64, 32),
            nn.Conv2d(32, 11, kernel_size=1)
        ])    
            
        self.layer_norm = nn.LayerNorm(initial_hidden_channels[-1]) if use_layer_norm else nn.Identity()
        
        self.size_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(initial_hidden_channels[-1], 128),
            nn.ELU() if self.use_elu else nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ELU() if self.use_elu else nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
            nn.Softplus()  # Ensures positive outputs
        )
           
        self.learning_rate = nn.Parameter(torch.tensor(0.0001))
        self.modification_rate = nn.Parameter(torch.tensor(0.1))
        self.content_weight = nn.Parameter(torch.tensor(1.0))
        self.size_weight = nn.Parameter(torch.tensor(1.0))
        self.lr_adjustment_factor = nn.Parameter(torch.tensor(1.0))
        

        self.performance_history = []
        self.modification_history = []
        self.feedback_factor = 1.0
        self.param_count_history = []
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        for layer in self.encoder:
            x = layer(x)
        
        # Dynamically adjust LayerNorm
        if self.use_layer_norm:
            # Create a new LayerNorm for the current feature map size
            self.layer_norm = nn.LayerNorm([x.size(2), x.size(3), x.size(1)]).to(x.device)
            # Apply LayerNorm
            x = self.layer_norm(x.permute(0, 2, 3, 1))
            x = x.permute(0, 3, 1, 2)
        
        size_pred = self.size_predictor(x)
        
        # Ensure predicted size is at least 1x1 and at most max_output_size x max_output_size
        pred_h = max(1, min(int(size_pred[0, 0].item()), self.max_output_size))
        pred_w = max(1, min(int(size_pred[0, 1].item()), self.max_output_size))
        
        for layer in self.decoder[:-1]:
            x = layer(x)
        
        content_pred = self.decoder[-1](x)
        
        # Ensure content_pred is the correct size
        if content_pred.shape[2:] != (pred_h, pred_w):
            content_pred = F.interpolate(content_pred, size=(pred_h, pred_w), mode='bilinear', align_corners=False)
        
        return content_pred, (pred_h, pred_w)

    def modify(self, performance_metric):
        modifications = []
        self.performance_history.append(performance_metric)
        
        if len(self.performance_history) > 1:
            performance_change = self.performance_history[-2] - self.performance_history[-1]
            
            # Determine learning rate change
            if performance_change > 0:  # Performance improved
                lr_change = min(performance_change * 0.1, 0.1)  # Increase LR, max 10% increase
            else:  # Performance worsened or stayed the same
                lr_change = max(performance_change * 0.1, -0.1)  # Decrease LR, max 10% decrease
            
            current_lr = self.learning_rate.item()
            new_lr = max(min(current_lr * (1 + lr_change), 1e-3), 1e-6)
            self.learning_rate.data = torch.tensor(new_lr)
            modifications.append(f"Updated learning rate: {new_lr:.6f}")
        
        if self.allow_architecture_mod:
            modifications.extend(self._modify_architecture())

        for param_name in ['modification_rate', 'content_weight', 'size_weight']:
            param = getattr(self, param_name)
            change = param.item() * 0.1 * (torch.rand(1).item() - 0.5) * self.feedback_factor
            new_value = max(0, param.item() + change)
            param.data = torch.tensor(new_value, device=param.device, dtype=param.dtype)
            modifications.append(f"Updated {param_name}: {new_value:.4f}")
        
        self.modification_history.append(modifications)
        return modifications

    def _update_feedback_factor(self):
        if len(self.performance_history) > 1:
            performance_change = self.performance_history[-2] - self.performance_history[-1]
            if performance_change > 0:
                self.feedback_factor = min(2.0, self.feedback_factor * 1.1)
            else:
                self.feedback_factor = max(0.5, self.feedback_factor * 0.9)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _modify_architecture(self):
        modifications = []
        current_param_count = self.count_parameters()
        
        # Analyze recent performance trend
        if len(self.performance_history) >= 5:
            recent_trend = sum(self.performance_history[-5:]) / 5
            long_term_trend = sum(self.performance_history) / len(self.performance_history)
            
            if recent_trend < long_term_trend:  # Performance is improving
                target_param_count = int(current_param_count * 1.1)  # Increase by 10%
            else:  # Performance is stagnating or worsening
                target_param_count = int(current_param_count * 0.9)  # Decrease by 10%
        else:
            target_param_count = current_param_count  # Not enough history, maintain current size
        
        for module_list in [self.encoder, self.decoder[:-1]]:
            for i, layer in enumerate(module_list):
                if isinstance(layer, DynamicLayer) and torch.rand(1).item() < self.modification_rate:
                    current_size = layer.conv.out_channels
                    if self.count_parameters() < target_param_count:
                        new_size = min(current_size * 2, current_size + 64)  # Increase, but not more than double
                    else:
                        new_size = max(current_size // 2, current_size - 64, 8)  # Decrease, but not less than half or 8
                    
                    new_size = new_size - (new_size % 8)  # Ensure it's divisible by 8
                    if new_size != current_size:
                        new_layer = DynamicLayer(layer.conv.in_channels, new_size)
                        with torch.no_grad():
                            min_channels = min(current_size, new_size)
                            new_layer.conv.weight[:min_channels] = layer.conv.weight[:min_channels]
                            new_layer.conv.bias[:min_channels] = layer.conv.bias[:min_channels]
                        module_list[i] = new_layer
                        
                        # Update the next layer's input channels
                        if i < len(module_list) - 1:
                            next_layer = module_list[i+1]
                            if isinstance(next_layer, DynamicLayer):
                                next_layer.conv = nn.Conv2d(new_size, next_layer.conv.out_channels, 
                                                            kernel_size=next_layer.conv.kernel_size, 
                                                            padding=next_layer.conv.padding)
                                nn.init.kaiming_normal_(next_layer.conv.weight, mode='fan_out', nonlinearity='relu')
                                nn.init.zeros_(next_layer.conv.bias)
                        
                        modifications.append(f"Modified layer size: {current_size} -> {new_size}")
        
        self.param_count_history.append(self.count_parameters())
        return modifications

    def get_architecture_summary(self):
        summary = []
        summary.append("Encoder:")
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, DynamicLayer):
                summary.append(f"  Layer {i}: Conv2d(in={layer.conv.in_channels}, out={layer.conv.out_channels})")
        
        summary.append("Decoder:")
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, DynamicLayer):
                summary.append(f"  Layer {i}: Conv2d(in={layer.conv.in_channels}, out={layer.conv.out_channels})")
            elif isinstance(layer, nn.Conv2d):
                summary.append(f"  Layer {i}: Conv2d(in={layer.in_channels}, out={layer.out_channels})")
        
        return summary
    
    def adjust_learning_rate(self, optimizer, current_loss, previous_loss=None, min_lr=1e-6, max_lr=1e-3):
        if previous_loss is None:
            return  # No adjustment on first iteration

        # Calculate relative change in loss
        relative_change = (previous_loss - current_loss) / previous_loss

        # Adjust learning rate based on performance
        adjustment_factor = 1.0
        if relative_change > 0.01:  # Loss decreased significantly
            adjustment_factor = 1.1  # Increase LR by 10%
        elif relative_change < -0.01:  # Loss increased significantly
            adjustment_factor = 0.9  # Decrease LR by 10%

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            new_lr = max(min(current_lr * adjustment_factor, max_lr), min_lr)
            param_group['lr'] = new_lr

        self.learning_rate.data = torch.tensor(optimizer.param_groups[0]['lr'])
        
        return adjustment_factor

    def analyze_gradient_flow(self):
        total_grad = 0
        for layer in self.encoder + self.decoder:
            if isinstance(layer, DynamicLayer):
                layer_grad = layer.conv.weight.grad.abs().mean().item()
                total_grad += layer_grad
        return total_grad / (len(self.encoder) + len(self.decoder))

    def analyze_layer_importance(self):
        importances = []
        for layer in self.encoder + self.decoder:
            if isinstance(layer, DynamicLayer):
                importance = torch.norm(layer.conv.weight)
                importances.append(importance.item())
        return importances
    
    def get_learning_rate(self):
        return self.learning_rate.item()