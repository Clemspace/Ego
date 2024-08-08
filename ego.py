import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np

class AdaptiveModificationScheduler:
    def __init__(self, initial_frequency=0.2, min_frequency=0.05, max_frequency=0.5, 
                 cooldown_period=5, performance_window=20, threshold=0.05):
        self.current_frequency = initial_frequency
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.cooldown_period = cooldown_period
        self.performance_window = performance_window
        self.performance_history = []
        self.last_modification_epoch = 0
        self.threshold = threshold
        self.frequency_history = []

    def should_modify(self, current_epoch):
        if current_epoch - self.last_modification_epoch < self.cooldown_period:
            return False
        return random.random() < self.current_frequency

    def update_frequency(self, current_performance):
        self.performance_history.append(current_performance)
        
        if len(self.performance_history) > self.performance_window:
            self.performance_history.pop(0)
            
            recent_performance = self.performance_history[-5:]
            overall_performance = self.performance_history
            
            recent_trend = np.mean(recent_performance)
            overall_trend = np.mean(overall_performance)
            
            # Calculate the rate of change
            if len(recent_performance) > 1:
                rate_of_change = (recent_performance[-1] - recent_performance[0]) / len(recent_performance)
            else:
                rate_of_change = 0
            
            # Adjust frequency based on performance trends and rate of change
            if recent_trend < overall_trend and rate_of_change < 0:
                # Performance is improving and loss is decreasing
                self.current_frequency = min(self.current_frequency * 1.1, self.max_frequency)
            elif recent_trend > overall_trend and rate_of_change > self.threshold:
                # Performance is worsening and loss is increasing too quickly
                self.current_frequency = max(self.current_frequency * 0.5, self.min_frequency)
            elif abs(rate_of_change) < self.threshold:
                # Performance is relatively stable
                self.current_frequency = max(self.current_frequency * 0.95, self.min_frequency)
            
            self.frequency_history.append(self.current_frequency)

    def record_modification(self, epoch):
        self.last_modification_epoch = epoch

    def get_frequency_stats(self):
        if not self.frequency_history:
            return None, None
        return np.mean(self.frequency_history), np.std(self.frequency_history)

class SelfModifyingNetwork(nn.Module):
    def __init__(self, input_channels=10, initial_hidden_channels=[64, 128], max_size_mb=50,
                 enable_layer_addition=True, enable_activation_modification=True, 
                 enable_forward_modification=True):
        super(SelfModifyingNetwork, self).__init__()
        self._max_size_mb = max_size_mb

        self.enable_layer_addition = enable_layer_addition
        self.enable_activation_modification = enable_activation_modification
        self.enable_forward_modification = enable_forward_modification

        self.onehot = nn.Embedding(input_channels, input_channels)
        self.onehot.weight.data = torch.eye(input_channels)
        self.onehot.weight.requires_grad = False

        self.encoder = nn.ModuleList([
            nn.Conv2d(input_channels, initial_hidden_channels[0], kernel_size=3, padding=1),
            nn.Conv2d(initial_hidden_channels[0], initial_hidden_channels[1], kernel_size=3, padding=1),
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.decoder = nn.ModuleList([
            nn.Conv2d(initial_hidden_channels[-1], 64, kernel_size=3, padding=1),
            nn.Conv2d(64, input_channels, kernel_size=1)
        ])

        self.modification_history = []
        self.performance_history = []
        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
        })
        self.current_activation = 'relu'

    @property
    def max_size_mb(self):
        return self._max_size_mb

    def forward(self, x):
        original_size = (x.shape[1], x.shape[2])
        
        x = self.onehot(x.long())
        x = x.permute(0, 3, 1, 2).float()
        
        for layer in self.encoder:
            x = self.activations[self.current_activation](layer(x))
        
        x = self.adaptive_pool(x)
        
        for layer in self.decoder[:-1]:
            x = self.activations[self.current_activation](layer(x))
        
        x = self.decoder[-1](x)
        
        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        return x

    def modify(self, performance_metric):
        self.performance_history.append(performance_metric)
        modifications = []

        if self.enable_layer_addition and self.get_model_size() < self.max_size_mb * 0.9:
            if random.random() < 0.2:  # 20% chance to add a new layer
                new_channels = self.encoder[-1].out_channels
                new_layer = nn.Conv2d(new_channels, new_channels, kernel_size=3, padding=1)
                self.encoder.append(new_layer)
                modifications.append(f"Added new encoder layer: Conv2d({new_channels}, {new_channels}, kernel_size=3)")

        if self.enable_activation_modification:
            if random.random() < 0.1:
                new_activation = random.choice(list(self.activations.keys()))
                if new_activation != self.current_activation:
                    self.current_activation = new_activation
                    modifications.append(f"Changed activation to: {self.current_activation}")

        if self.enable_forward_modification:
            if len(self.performance_history) > 5:
                recent_trend = sum(self.performance_history[-5:]) / 5
                if recent_trend > sum(self.performance_history) / len(self.performance_history):
                    if not hasattr(self, 'dropout'):
                        self.dropout = nn.Dropout(0.1)
                        modifications.append("Added dropout to forward pass")

        # Check if model size exceeds limit after modifications
        if self.get_model_size() > self.max_size_mb:
            # Revert the last modification
            if "Added new encoder layer" in modifications[-1]:
                self.encoder.pop()
                modifications.pop()
                modifications.append("Reverted layer addition due to size limit")

        self.modification_history.extend(modifications)
        return modifications

    def get_model_size(self):
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def __setattr__(self, name, value):
        if name == 'max_size_mb':
            raise AttributeError("'max_size_mb' is read-only and cannot be modified.")
        super().__setattr__(name, value)